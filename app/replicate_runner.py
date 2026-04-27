"""Replicate-backed LivePortrait runner.

Swaps in for liveportrait_runner when USE_REPLICATE=1. Keeps the same
(source_image_path, preset_key) -> list[PIL.Image] contract, so the Flask
handler doesn't care which backend produced the frames.

Each preset maps to a pre-recorded driving video under app/driving_videos/.
The video drives the portrait via Replicate's fofr/live-portrait model; the
result is an mp4 we decode to frames and hand to gif_writer.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import replicate
import requests
from dotenv import load_dotenv
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
DRIVING_DIR = APP_DIR / "driving_videos"
# Small on-disk cache so the driving-video upload URLs survive app restarts.
# Replicate's Files API keeps files for ~24h by default; we re-upload on miss.
UPLOAD_CACHE = APP_DIR / ".driving_video_urls.json"

load_dotenv(APP_DIR / ".env")

# Pinned model version. Using owner/model alone returns 404 from some
# Replicate SDK paths; owner/model:sha hits the stable /predictions endpoint.
# Version sha taken from `print_schema()`; bump when the model updates.
MODEL = "fofr/live-portrait:067dd98cc3e5cb396c4a9efb4bba3eec6c4a9d271211325c477518fc6485e146"

# Input field names — these match the fofr/live-portrait schema. If Replicate
# rejects the call with "unexpected input ...", run `print_schema()` below to
# see the real field names and update these two constants.
INPUT_FACE = "face_image"
INPUT_DRIVING = "driving_video"


def _client() -> replicate.Client:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN is not set. Put it in app/.env or export it."
        )
    return replicate.Client(api_token=token)


def print_schema() -> None:
    """Diagnostic: print the live model schema so we can verify field names."""
    client = _client()
    model = client.models.get(MODEL)
    version = model.latest_version
    print("Model:", MODEL, "version:", version.id)
    print("Inputs:")
    for name, schema in version.openapi_schema["components"]["schemas"]["Input"]["properties"].items():
        print(f"  - {name}: {schema.get('type', '?')}  {schema.get('description', '')}")


def _driving_video_for(preset_key: str) -> Path:
    path = DRIVING_DIR / f"{preset_key}.mp4"
    if not path.exists():
        raise FileNotFoundError(
            f"No driving video for preset '{preset_key}'. Expected: {path}"
        )
    return path


def _download(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


_upload_lock = threading.Lock()


def _load_upload_cache() -> dict[str, dict]:
    if UPLOAD_CACHE.exists():
        try:
            return json.loads(UPLOAD_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_upload_cache(cache: dict[str, dict]) -> None:
    UPLOAD_CACHE.write_text(json.dumps(cache, indent=2))


def _driving_video_url(client: replicate.Client, preset_key: str) -> str:
    """Upload the driving video to Replicate once and cache the URL on disk.

    Replicate re-uploads file inputs on every `client.run()` when given a
    file object, which adds 30-90s for an 8MB clip on residential wifi.
    The Files API lets us stash it server-side once and pass a URL instead.
    """
    driving_path = _driving_video_for(preset_key)
    path_key = str(driving_path.resolve())
    mtime = driving_path.stat().st_mtime

    with _upload_lock:
        cache = _load_upload_cache()
        entry = cache.get(path_key)
        # Re-upload if the file changed on disk (user replaced the video).
        if entry and entry.get("mtime") == mtime and entry.get("url"):
            return entry["url"]

        with open(driving_path, "rb") as f:
            file_obj = client.files.create(file=f)
        # Newer SDK returns a FileOutput with .urls.get or a dict-like urls map.
        url = getattr(file_obj, "urls", {}).get("get") if hasattr(file_obj, "urls") else None
        if url is None and hasattr(file_obj, "url"):
            url = file_obj.url
        if url is None:
            raise RuntimeError(
                f"Couldn't get URL from Replicate Files API response: {file_obj!r}"
            )

        cache[path_key] = {"url": url, "mtime": mtime}
        _save_upload_cache(cache)
        return url


def prewarm_uploads() -> None:
    """Upload every driving video up-front. Call once at app startup to make
    the first /generate request fast."""
    client = _client()
    for path in sorted(DRIVING_DIR.glob("*.mp4")):
        preset = path.stem
        print(f"[replicate] uploading driving video '{preset}'...", end=" ", flush=True)
        url = _driving_video_url(client, preset)
        print("ok")


def _invalidate_cached_url(preset_key: str) -> None:
    driving_path = _driving_video_for(preset_key)
    path_key = str(driving_path.resolve())
    with _upload_lock:
        cache = _load_upload_cache()
        if path_key in cache:
            del cache[path_key]
            _save_upload_cache(cache)


def _run_with_retry(client, input_dict, preset_key):
    """Run the model; on 404 (expired Files URL), invalidate cache and retry once."""
    try:
        return client.run(MODEL, input=input_dict)
    except replicate.exceptions.ModelError as e:
        if "404" not in str(e) or "/v1/files/" not in str(e):
            raise
        print(f"[replicate] driving-video URL expired for '{preset_key}' — re-uploading and retrying.")
        _invalidate_cached_url(preset_key)
        fresh_url = _driving_video_url(client, preset_key)
        input_dict[INPUT_DRIVING] = fresh_url
        return client.run(MODEL, input=input_dict)


def animate(source_image_path: str, preset_key: str) -> list[Image.Image]:
    """Animate `source_image_path` using the driving video for `preset_key`."""
    import time
    t0 = time.perf_counter()

    client = _client()
    driving_url = _driving_video_url(client, preset_key)
    t_upload = time.perf_counter()

    with open(source_image_path, "rb") as src_f:
        output = _run_with_retry(
            client,
            {
                INPUT_FACE: src_f,
                INPUT_DRIVING: driving_url,
                # Cap to ~54 frames — matches gif_writer's 3s@18fps target and
                # avoids slow CPU-bound GIF encoding of 120+ frames locally.
                "video_frame_load_cap": 54,
            },
            preset_key,
        )
    t_inference = time.perf_counter()

    if isinstance(output, list):
        output = output[0]
    if hasattr(output, "read"):
        video_bytes = output.read()
    elif isinstance(output, str):
        video_bytes = _download(output)
    else:
        raise RuntimeError(f"Unexpected Replicate output type: {type(output)}")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        arr_frames = list(iio.imiter(tmp.name))
    t_decode = time.perf_counter()

    print(
        f"[replicate] preset={preset_key} "
        f"upload_url={t_upload - t0:.1f}s "
        f"inference+queue={t_inference - t_upload:.1f}s "
        f"decode={t_decode - t_inference:.1f}s "
        f"frames={len(arr_frames)}"
    )

    return [Image.fromarray(f) for f in arr_frames]


def using_cuda() -> Optional[bool]:
    # Runs on Replicate's L40S; "cuda" from this app's perspective is moot.
    return None
