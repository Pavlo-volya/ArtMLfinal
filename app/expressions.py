"""Expression preset parameter trajectories.

Each preset returns a list of dicts, one per frame. Each dict is a set of
retargeting scalars handed to LivePortrait's retargeting module:
    rotate_pitch, rotate_yaw, rotate_roll  (head pose, degrees)
    eye_close   (0.0 = open, 1.0 = fully closed)
    lip_open    (0.0 = closed, ~0.3 = parted)
    smile       (0.0 = neutral, ~0.6 = smiling)

Frames LOOP: first frame == last frame (closed cycle).
Default frame count targets 3-second loops at 18 fps (see gif_writer.TARGET_FPS).
"""

from __future__ import annotations

import math
from typing import Callable

FRAMES = 54  # 3.0 s @ 18 fps


def _ease(t: float) -> float:
    """Cosine ease-in-out, t in [0, 1] -> [0, 1]."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _loop_shape(n_frames: int, peak: float) -> list[float]:
    """Values that start at 0, rise to `peak`, and return to 0 smoothly."""
    out = []
    for i in range(n_frames):
        t = i / n_frames
        if t < 0.5:
            out.append(peak * _ease(t * 2))
        else:
            out.append(peak * _ease((1.0 - t) * 2))
    return out


def _blink_curve(n_frames: int, blink_duration: int = 10, peak: float = 1.0) -> list[float]:
    """Single blink centered in the cycle; `blink_duration` frames wide, rises to `peak` at midpoint."""
    eye = [0.0] * n_frames
    start = n_frames // 2 - blink_duration // 2
    for i in range(blink_duration):
        t = i / blink_duration
        # Triangle: 0 -> peak -> 0
        eye[start + i] = peak * ((t * 2) if t < 0.5 else ((1.0 - t) * 2))
    return eye


def preset_smile(n_frames: int = FRAMES) -> list[dict]:
    smile_vals = _loop_shape(n_frames, peak=0.6)
    lip_vals = _loop_shape(n_frames, peak=0.08)
    return [
        {
            "rotate_pitch": 0.0,
            "rotate_yaw": 0.0,
            "rotate_roll": 0.0,
            "eye_close": 0.0,
            "lip_open": lip_vals[i],
            "smile": smile_vals[i],
        }
        for i in range(n_frames)
    ]


def preset_blink(n_frames: int = FRAMES) -> list[dict]:
    # Single-eye wink (uses LivePortrait's asymmetric wink expression basis,
    # not the symmetric retarget_eye head). Wink magnitude is larger than the
    # eye_close ratio — the wink basis deltas are small in absolute value.
    wink = _blink_curve(n_frames, blink_duration=12, peak=2.5)
    return [
        {
            "rotate_pitch": 0.0,
            "rotate_yaw": 0.0,
            "rotate_roll": 0.0,
            "eye_close": 0.0,
            "wink": wink[i],
            "lip_open": 0.0,
            "smile": 0.0,
        }
        for i in range(n_frames)
    ]


def preset_smile_blink(n_frames: int = FRAMES) -> list[dict]:
    base = preset_smile(n_frames)
    wink = _blink_curve(n_frames, blink_duration=12, peak=2.5)
    for i in range(n_frames):
        base[i]["wink"] = wink[i]
    return base


def preset_head_tilt(n_frames: int = FRAMES) -> list[dict]:
    # Gentle side-to-side tilt: one full sine cycle across the loop.
    out = []
    for i in range(n_frames):
        phase = 2 * math.pi * i / n_frames
        yaw = 6.0 * math.sin(phase)
        roll = 2.0 * math.sin(phase)
        pitch = -1.5 * math.cos(phase)
        out.append({
            "rotate_pitch": pitch,
            "rotate_yaw": yaw,
            "rotate_roll": roll,
            "eye_close": 0.0,
            "lip_open": 0.0,
            "smile": 0.0,
        })
    return out


def preset_head_tilt_smile(n_frames: int = FRAMES) -> list[dict]:
    tilt = preset_head_tilt(n_frames)
    smile = preset_smile(n_frames)
    for i in range(n_frames):
        tilt[i]["smile"] = smile[i]["smile"]
        tilt[i]["lip_open"] = smile[i]["lip_open"]
    return tilt


PRESETS: dict[str, Callable[[], list[dict]]] = {
    "smile": preset_smile,
    "blink": preset_blink,
    "smile_blink": preset_smile_blink,
    "head_tilt": preset_head_tilt,
    "head_tilt_smile": preset_head_tilt_smile,
}

PRESET_LABELS = {
    "smile": "Smile",
    "blink": "Blink",
    "smile_blink": "Smile + Blink",
    "head_tilt": "Gentle Head Tilt",
    "head_tilt_smile": "Head Tilt + Smile",
}


def get_frames(preset_key: str) -> list[dict]:
    if preset_key not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_key}")
    return PRESETS[preset_key]()
