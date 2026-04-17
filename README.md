# Portrait Animator

A local Flask web app that animates a portrait photo using
[LivePortrait](https://github.com/KwaiVGI/LivePortrait)'s retargeting module
and exports a looping, phone-wallpaper-sized GIF. A QR code links directly to
the download URL so you can scan it onto your phone.

## Features

- Drag-and-drop portrait upload
- 5 expression presets: Smile, Blink, Smile + Blink, Gentle Head Tilt, Head Tilt + Smile
- Per-preset parameter trajectories drive LivePortrait's retargeting scalars
  (rotate_pitch/yaw/roll, eye_close, lip_open, smile)
- Output: looping GIF at 1080×1920 (phone portrait), 24 fps, per-frame 256-color
  optimized palette with Floyd-Steinberg dithering
- Auto-downscales if the GIF would exceed 15 MB
- QR code linking to `http://localhost:5001/download/<file_id>`
- CUDA GPU if available; CPU fallback otherwise

## Project layout

```
ArtMLfinal/
├── README.md
└── app/
    ├── app.py                 # Flask backend
    ├── expressions.py         # Preset parameter trajectories
    ├── liveportrait_runner.py # LivePortrait wrapper
    ├── gif_writer.py          # High-quality looping GIF encoder
    ├── requirements.txt
    ├── setup.sh
    ├── liveportrait/          # (cloned by setup.sh)
    ├── static/
    │   ├── uploads/           # Uploaded source images
    │   └── outputs/           # Generated GIFs
    └── templates/
        └── index.html
```

## Setup

Requires Python **3.10 or 3.11** (LivePortrait's pinned deps don't have
prebuilt wheels for 3.12+ on all platforms, which forces source builds
for scipy/numpy) and `git`. A CUDA GPU is strongly recommended; CPU
inference will still work but is slow (minutes per clip).

On macOS arm64 / Apple Silicon, install Python 3.11 via `brew install python@3.11`
first if you don't have it.

```bash
cd app
bash setup.sh
```

`setup.sh` will:

1. Clone LivePortrait into `app/liveportrait/`
2. Create a Python venv at `app/venv/`
3. Install `requirements.txt` plus LivePortrait's own requirements
4. Download pretrained weights from HuggingFace
   (`KwaiVGI/LivePortrait`) into `app/liveportrait/pretrained_weights/`

## Run

```bash
cd app
source venv/bin/activate
python app.py
```

Open <http://localhost:5001> in your browser.

## Using the app

1. Drop a clear, front-facing portrait photo onto the upload area
2. Pick an expression from the dropdown
3. Click **Generate** (first run warms up the models; subsequent runs are faster)
4. Preview the animation, scan the QR code with your phone, tap to download

## Notes on the LivePortrait integration

- We use LivePortrait's **retargeting** path (not driving-video): parameter
  scalars are pushed directly into the keypoint / expression basis per frame.
- The source portrait's features are extracted **once** per request and reused
  for every frame, so animation cost scales linearly with frame count.
- The `flag_stitching` and `flag_pasteback` flags are enabled so the animated
  face is blended cleanly back onto the original photo background.

## Tweaking presets

Edit `app/expressions.py`. Each preset returns a list of dicts, one per frame,
with `rotate_pitch`, `rotate_yaw`, `rotate_roll`, `eye_close`, `lip_open`,
`smile` scalars. Frames loop by construction (first ≈ last).

## Troubleshooting

- **"Face not detected"**: try a sharper, front-facing photo; avoid heavy
  occlusion (sunglasses, hands on face).
- **OOM on GPU**: lower frame count in `expressions.py`, or set
  `flag_use_half_precision=True` (already default on CUDA).
- **CPU run is slow**: expect 1–3 min per clip on modern CPUs; reduce
  `n_frames` in presets for a faster test.
- **GIF too large / low quality**: `gif_writer.py` auto-steps down resolution
  until the file fits under 15 MB. Adjust `MAX_SIZE_BYTES` there if needed.
- **`pip` tries to build `scipy` from source (meson error)**: you're on
  Python 3.12+ or a platform without a matching wheel. Re-run `setup.sh`
  after installing Python 3.11 (`brew install python@3.11`); setup.sh
  auto-picks it. If you already have a broken venv, `rm -rf app/venv` first.
