# Still With Us

A local Flask web app that turns a portrait photo into a gently looping phone-wallpaper GIF. Built around remembering grandparents — warm light-blue UI, cozy loading messages, the result framed inside iPhone and Galaxy mockups so you can see how it'll look on a phone.

The animation runs on **Replicate** (`fofr/live-portrait`, an Nvidia L40S in their cloud). You don't need a local GPU and you don't need PyTorch installed — your machine just runs Flask and encodes the GIF.

## Features

- Drag-and-drop portrait upload
- 3 expression presets — **Smile**, **Blink** (single-eye wink), **Smile + Blink**
- Each preset is driven by a short pre-recorded driving video (`app/driving_videos/*.mp4`)
- Looping GIF output at 540×960, 18 fps, 256-color per-frame palette with Floyd–Steinberg dithering
- Result preview rendered inside an iPhone and a Galaxy phone frame
- Driving videos are uploaded to Replicate **once** at startup and the URLs cached on disk, so each `/generate` only ships the source image
- Auto-retry on expired Replicate Files URLs (cache self-heals)

## Project layout

```
ArtMLfinal/
├── README.md
└── app/
    ├── app.py                  # Flask backend
    ├── replicate_runner.py     # Replicate API wrapper (the runner used in this mode)
    ├── gif_writer.py           # GIF encoder
    ├── expressions.py          # Preset names/labels (the scalar trajectories
    │                           # are unused in Replicate mode but kept for the
    │                           # local fallback)
    ├── requirements.txt
    ├── .env.example            # Copy to .env and fill in your token
    ├── driving_videos/         # smile.mp4, blink.mp4, smile_blink.mp4
    ├── static/
    │   ├── uploads/            # Uploaded source images
    │   └── outputs/            # Generated GIFs
    └── templates/
        └── index.html
```

## Setup

Requires Python **3.10 or 3.11**, `ffmpeg` on PATH (`brew install ffmpeg` on macOS), and a free [Replicate](https://replicate.com) account with a few dollars of prepaid credit (~$0.04/run).

```bash
cd app
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: paste your token (https://replicate.com/account/api-tokens)
#            and keep USE_REPLICATE=1
```

That's it — no LivePortrait clone, no model weights, no GPU.

## Run

```bash
cd app
source venv/bin/activate
python app.py
```

On first start, the app uploads the 3 driving videos to Replicate's Files API (~10–15 s) and caches the URLs in `app/.driving_video_urls.json`. Subsequent starts skip this. Open <http://localhost:5001>.

## Using the app

1. Drop a clear front-facing portrait into the upload area
2. Pick an expression
3. Click **Generate** — a cozy loader plays through messages like *"Pulling out the old photo album…"* while Replicate works
4. The result appears framed inside an iPhone and a Galaxy mockup; click **Download GIF** to save

## How a request flows

```
browser ─upload─▶ Flask /generate
                    │
                    ├─ POST source image + cached driving-video URL
                    │   to fofr/live-portrait on Replicate (L40S)
                    │
                    │   (Replicate returns mp4 URL or FileOutput)
                    ▼
                 imageio + ffmpeg decode mp4 → PIL frames
                    │
                    ▼
                 gif_writer: fit-cover to 540×960,
                 quantize per-frame (256-color + Floyd–Steinberg),
                 write looping GIF
                    │
                    ▼
                 JSON {gif_url, download_url}
```

## Tools used in this mode

**Backend:** Flask · Jinja2 · Werkzeug · Pillow · imageio · imageio-ffmpeg · ffmpeg (system) · replicate · requests · python-dotenv

**External service:** Replicate — runs `fofr/live-portrait` on an Nvidia L40S (the model itself is PyTorch + LivePortrait + InsightFace, but those run on Replicate's machines, not yours)

**Frontend:** vanilla HTML / CSS / JavaScript (no framework)

## Performance & cost

- **Cost:** ~$0.02–0.04 per generation (metered by Replicate, deducted from prepaid credit)
- **Warm:** ~25–40 s end-to-end (queue+inference ~15–30 s, source upload + decode + GIF encode ~10–15 s)
- **Cold:** add ~60–90 s if no requests have hit the model in the last few minutes — Replicate spins workers down when idle. Click Generate once before a demo to warm it up.

## Troubleshooting

- **`No ffmpeg exe could be found`** — install ffmpeg (`brew install ffmpeg`) or `pip install --force-reinstall imageio-ffmpeg`.
- **`REPLICATE_API_TOKEN is not set`** — copy `.env.example` to `.env` and paste your token.
- **`404 Client Error` on a `/v1/files/...` URL** — the cached driving-video URL expired (~24 h TTL). The runner already auto-invalidates and retries; if it still fails, delete `app/.driving_video_urls.json` and restart Flask to re-upload.
- **GIF generation slow (>1 min)** — Replicate cold-start, not your code. Check the per-call timing logged in the terminal (`[replicate] preset=… inference+queue=… [gif_writer] took …s`).
- **Nothing happens / output looks frozen** — make sure your driving videos in `app/driving_videos/` show *visible*, slightly exaggerated motion. The model copies the magnitude of motion from the driving video; subtle = subtle.

## Switching to local execution

The repo also contains a local PyTorch path (`liveportrait_runner.py`) that runs LivePortrait directly on your machine. To use it, set `USE_REPLICATE=0` in `.env` and run `bash setup.sh` to clone LivePortrait and download weights. Not covered in this README.
