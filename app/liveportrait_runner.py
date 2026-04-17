"""Thin wrapper around the LivePortrait retargeting pipeline.

Loads LivePortrait's models once and exposes `animate()` — takes a source
image plus a list of per-frame retargeting scalars, returns a list of RGB
PIL Images at the source's full resolution.

LivePortrait is expected to be cloned under ./liveportrait/ (see setup.sh).
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional

# Must be set BEFORE torch imports: allow MPS to CPU-fallback for ops it
# doesn't implement (e.g. aten::grid_sampler_3d used by the warping module).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_LP_DIR = os.path.join(_APP_DIR, "liveportrait")

if _LP_DIR not in sys.path:
    sys.path.insert(0, _LP_DIR)


_pipeline_lock = threading.Lock()
_pipeline = None
_use_cuda: Optional[bool] = None


def _detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on what's available."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_pipeline():
    """Initialize LivePortrait once. Thread-safe."""
    global _pipeline, _use_cuda
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        from src.config.crop_config import CropConfig
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_pipeline import LivePortraitPipeline

        device = _detect_device()
        _use_cuda = (device == "cuda")
        # Only force CPU when we truly have nothing else. MPS is a huge win
        # over CPU on Apple Silicon (~10× for the warping+SPADE generator).
        flag_force_cpu = (device == "cpu")
        device_id = 0 if _use_cuda else -1

        inference_cfg = InferenceConfig(
            # Half precision is safe on CUDA; on MPS keep fp32 for stability.
            flag_use_half_precision=_use_cuda,
            flag_crop_driving_video=False,
            device_id=device_id,
            flag_force_cpu=flag_force_cpu,
            flag_stitching=True,
            flag_relative_motion=True,
            flag_pasteback=True,
            flag_do_crop=True,
        )
        crop_cfg = CropConfig(device_id=device_id, flag_force_cpu=flag_force_cpu)

        _pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg, crop_cfg=crop_cfg
        )
        return _pipeline


def _prepare_source(pipeline, source_image_path: str):
    """Extract identity/appearance features from the source portrait once."""
    import cv2
    from src.utils.camera import get_rotation_matrix
    from src.utils.crop import prepare_paste_back
    from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio

    wrapper = pipeline.live_portrait_wrapper
    cropper = pipeline.cropper
    inference_cfg = wrapper.inference_cfg

    img_bgr = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image at {source_image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    crop_info = cropper.crop_source_image(img_rgb, crop_cfg=cropper.crop_cfg)
    if crop_info is None or "img_crop_256x256" not in crop_info:
        raise RuntimeError(
            "No face detected in the uploaded image. "
            "Try a clear, front-facing portrait."
        )

    I_s = wrapper.prepare_source(crop_info["img_crop_256x256"])
    x_s_info = wrapper.get_kp_info(I_s)
    R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
    f_s = wrapper.extract_feature_3d(I_s)
    x_s = wrapper.transform_keypoint(x_s_info)

    # Source ratios (used as baselines for eye/lip retargeting targets).
    source_lmk = crop_info["lmk_crop"]
    src_eye_ratio = float(calc_eye_close_ratio(source_lmk[None]).mean())
    src_lip_ratio = float(calc_lip_close_ratio(source_lmk[None])[0][0])

    mask_ori = prepare_paste_back(
        inference_cfg.mask_crop,
        crop_info["M_c2o"],
        dsize=(img_rgb.shape[1], img_rgb.shape[0]),
    )

    return {
        "img_rgb": img_rgb,
        "crop_info": crop_info,
        "source_lmk": source_lmk,
        "x_s_info": x_s_info,
        "R_s": R_s,
        "f_s": f_s,
        "x_s": x_s,
        "mask_ori": mask_ori,
        "src_eye_ratio": src_eye_ratio,
        "src_lip_ratio": src_lip_ratio,
    }


def _apply_smile(delta_new, smile: float):
    """Port of GradioPipeline.update_delta_new_smile — in-place on delta_new."""
    delta_new[0, 20, 1] += smile * -0.01
    delta_new[0, 14, 1] += smile * -0.02
    delta_new[0, 17, 1] += smile * 0.0065
    delta_new[0, 17, 2] += smile * 0.003
    delta_new[0, 13, 1] += smile * -0.00275
    delta_new[0, 16, 1] += smile * -0.00275
    delta_new[0, 3, 1] += smile * -0.0035
    delta_new[0, 7, 1] += smile * -0.0035
    return delta_new


def _apply_wink(delta_new, wink: float):
    """Port of GradioPipeline.update_delta_new_wink — asymmetric single-eye close."""
    delta_new[0, 11, 1] += wink * 0.001
    delta_new[0, 13, 1] += wink * -0.0003
    delta_new[0, 17, 0] += wink * 0.0003
    delta_new[0, 17, 1] += wink * 0.0003
    delta_new[0, 3, 1] += wink * -0.0003
    return delta_new


def _render_frame(pipeline, src, scalars: dict) -> np.ndarray:
    """Render a single animated frame, composited onto the original photo."""
    import torch
    from src.utils.camera import get_rotation_matrix
    from src.utils.crop import paste_back

    wrapper = pipeline.live_portrait_wrapper
    device = wrapper.device
    x_s_info = src["x_s_info"]
    x_s = src["x_s"].to(device)
    f_s = src["f_s"].to(device)
    R_s = src["R_s"].to(device)

    # Head rotation: deltas (degrees) relative to source pose.
    d_pitch = float(scalars.get("rotate_pitch", 0.0))
    d_yaw = float(scalars.get("rotate_yaw", 0.0))
    d_roll = float(scalars.get("rotate_roll", 0.0))
    R_d = get_rotation_matrix(
        x_s_info["pitch"] + d_pitch,
        x_s_info["yaw"] + d_yaw,
        x_s_info["roll"] + d_roll,
    ).to(device)

    # Relative rotation target (same formula GradioPipeline uses).
    R_d_new = (R_d @ R_s.permute(0, 2, 1)) @ R_s

    # Expression deltas: start from source expression and apply smile.
    x_c_s = x_s_info["kp"].to(device)
    delta_new = x_s_info["exp"].to(device).clone()
    scale_new = x_s_info["scale"].to(device)
    t_new = x_s_info["t"].to(device).clone()
    t_new[..., 2] = 0.0  # zero z-translation for clean pasteback

    smile = float(scalars.get("smile", 0.0))
    if smile != 0.0:
        delta_new = _apply_smile(delta_new, smile)
    wink = float(scalars.get("wink", 0.0))
    if wink != 0.0:
        delta_new = _apply_wink(delta_new, wink)

    x_d_new = scale_new * (x_c_s @ R_d_new + delta_new) + t_new

    # Eye close / lip open via retargeting heads — shift target ratio from source.
    eye_close = float(scalars.get("eye_close", 0.0))
    lip_open = float(scalars.get("lip_open", 0.0))

    if eye_close != 0.0:
        target_eye = max(0.0, src["src_eye_ratio"] * (1.0 - eye_close))
        combined_eye = wrapper.calc_combined_eye_ratio([[target_eye]], src["source_lmk"])
        eye_delta = wrapper.retarget_eye(x_s, combined_eye)
        x_d_new = x_d_new + eye_delta.reshape(x_d_new.shape)

    if lip_open != 0.0:
        target_lip = src["src_lip_ratio"] + lip_open
        combined_lip = wrapper.calc_combined_lip_ratio([[target_lip]], src["source_lmk"])
        lip_delta = wrapper.retarget_lip(x_s, combined_lip)
        x_d_new = x_d_new + lip_delta.reshape(x_d_new.shape)

    if wrapper.inference_cfg.flag_stitching:
        x_d_new = wrapper.stitching(x_s, x_d_new)

    out = wrapper.warp_decode(f_s, x_s, x_d_new)
    I_p = wrapper.parse_output(out["out"])[0]  # HxWx3 uint8, 512x512 crop

    composited = paste_back(
        I_p, src["crop_info"]["M_c2o"], src["img_rgb"], src["mask_ori"]
    )
    return composited


def animate(source_image_path: str, frame_params: list[dict]) -> list[Image.Image]:
    """Render a full animation.

    Args:
        source_image_path: path to the uploaded portrait.
        frame_params: list of per-frame scalar dicts (see expressions.py).

    Returns:
        List of PIL RGB Images at full source resolution.
    """
    import torch

    pipeline = _load_pipeline()
    src = _prepare_source(pipeline, source_image_path)

    frames: list[Image.Image] = []
    with torch.no_grad():
        for scalars in frame_params:
            rgb = _render_frame(pipeline, src, scalars)
            frames.append(Image.fromarray(rgb))
    return frames


def using_cuda() -> Optional[bool]:
    return _use_cuda
