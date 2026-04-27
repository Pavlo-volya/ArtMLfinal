"""Frame -> high-quality looping GIF at phone-wallpaper dimensions."""

from __future__ import annotations

import io
import os
from typing import Iterable

from PIL import Image

TARGET_W = 540    # Phone wallpapers are downscaled anyway; 540x960 is
TARGET_H = 960    # the sweet spot: ~4× fewer pixels than 1080x1920,
                  # encodes 5-6× faster, and visually indistinguishable
                  # once the GIF's 256-color palette kicks in.
TARGET_FPS = 18   # 18fps × 54 frames = 3.0s loop
MAX_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB (informational only, no retry)


def _fit_portrait(frame: Image.Image, w: int = TARGET_W, h: int = TARGET_H) -> Image.Image:
    """Cover-fit the frame into (w, h) preserving aspect ratio, centered."""
    src_w, src_h = frame.size
    src_ratio = src_w / src_h
    dst_ratio = w / h

    if src_ratio > dst_ratio:
        # source wider than target: scale by height, crop width
        new_h = h
        new_w = int(round(src_ratio * new_h))
    else:
        new_w = w
        new_h = int(round(new_w / src_ratio))
    resized = frame.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return resized.crop((left, top, left + w, top + h))


def _quantize_per_frame(frame: Image.Image) -> Image.Image:
    """256-color per-frame optimized palette with Floyd-Steinberg dithering."""
    return frame.convert("RGB").quantize(
        colors=256,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.FLOYDSTEINBERG,
    )


def _encode(frames: list[Image.Image], fps: int, out_path: str) -> int:
    """Encode frames to a looping GIF. Returns byte size."""
    duration_ms = int(round(1000.0 / fps))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,       # don't strip per-frame palettes
        duration=duration_ms,
        loop=0,               # infinite loop
        disposal=2,
    )
    return os.path.getsize(out_path)


def write_gif(frames: Iterable[Image.Image], out_path: str, fps: int = TARGET_FPS) -> dict:
    """Write frames to a looping GIF sized for a phone wallpaper.

    Single-pass: encoding GIFs is CPU-bound (quantize + Floyd-Steinberg
    dither per frame). Retrying at multiple resolutions was costing 60-80s
    per request. 540x960 comfortably fits under 15MB for typical portraits.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("No frames to encode")

    processed = [_quantize_per_frame(_fit_portrait(fr, TARGET_W, TARGET_H)) for fr in frames]
    size = _encode(processed, fps, out_path)
    return {
        "width": TARGET_W,
        "height": TARGET_H,
        "fps": fps,
        "bytes": size,
        "frames": len(processed),
    }
