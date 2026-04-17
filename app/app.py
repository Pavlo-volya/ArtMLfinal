"""Flask app: portrait upload -> LivePortrait animation -> looping GIF + QR."""

from __future__ import annotations

import io
import os
import socket
import uuid
from pathlib import Path

import qrcode
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from PIL import Image
from werkzeug.utils import secure_filename

import expressions
import gif_writer
import liveportrait_runner


def _lan_ip() -> str:
    """Best-effort LAN IP for this host. Falls back to 127.0.0.1."""
    # Force the env var to win if set.
    if os.environ.get("HOST_IP"):
        return os.environ["HOST_IP"]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually send — just makes the OS pick the outbound iface.
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "static" / "uploads"
OUTPUT_DIR = APP_DIR / "static" / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def _scan_url(file_id: str) -> str:
    """Always-scannable download URL — uses the LAN IP, not localhost."""
    port = int(os.environ.get("PORT", "5001"))
    return f"http://{_lan_ip()}:{port}/download/{file_id}"


@app.route("/")
def index():
    options = [
        {"key": k, "label": expressions.PRESET_LABELS[k]}
        for k in expressions.PRESETS.keys()
    ]
    return render_template("index.html", options=options)


@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    preset = request.form.get("preset", "smile")

    if preset not in expressions.PRESETS:
        return jsonify({"error": f"Unknown preset: {preset}"}), 400
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(secure_filename(file.filename))[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    file_id = uuid.uuid4().hex
    src_path = UPLOAD_DIR / f"{file_id}{ext}"
    file.save(src_path)

    frame_params = expressions.get_frames(preset)

    try:
        frames = liveportrait_runner.animate(str(src_path), frame_params)
    except Exception as e:
        app.logger.exception("Animation failed")
        return jsonify({"error": f"Animation failed: {e}"}), 500

    out_path = OUTPUT_DIR / f"{file_id}.gif"
    info = gif_writer.write_gif(frames, str(out_path))

    download_url = url_for("download", file_id=file_id)
    gif_url = url_for("static", filename=f"outputs/{file_id}.gif")

    return jsonify({
        "file_id": file_id,
        "gif_url": gif_url,
        "download_url": download_url,
        "info": info,
    })


@app.route("/download/<file_id>")
def download(file_id: str):
    file_id = secure_filename(file_id)
    path = OUTPUT_DIR / f"{file_id}.gif"
    if not path.exists():
        abort(404)
    return send_file(
        path,
        mimetype="image/gif",
        as_attachment=True,
        download_name=f"portrait_{file_id}.gif",
    )


@app.route("/qr/<file_id>.png")
@app.route("/qr/<file_id>")
def qr(file_id: str):
    file_id = secure_filename(file_id)
    if not (OUTPUT_DIR / f"{file_id}.gif").exists():
        abort(404)
    target = _scan_url(file_id)
    img = qrcode.make(target, box_size=10, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "cuda": liveportrait_runner.using_cuda(),
    })


if __name__ == "__main__":
    # Default to 5001 — macOS binds 5000 to AirPlay Receiver.
    port = int(os.environ.get("PORT", "5001"))
    lan = _lan_ip()
    print(f"\n  Local:   http://localhost:{port}")
    print(f"  Network: http://{lan}:{port}   <- QR / phone URL\n")
    # Single-process for model reuse; threaded for concurrent downloads/QR.
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
