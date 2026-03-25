"""
pipeline_utils.py – Shared data-pipeline helpers.

Provides utilities for:
  - Resolving model output directories
  - Pairing raw video files with their depth prediction files
  - Safe frame and mask loading
  - Depth map normalisation
  - Projected depth-from-distance calculations
  - Animal detection (MegaDetector fallback)
"""

import os
from glob import glob

import cv2
import numpy as np


# ── Supported file extensions ─────────────────────────────────────────────────

VIDEO_EXTS = (".mp4", ".avi")
DEPTH_EXTS = (".npy",)


# ── Model output directory resolution ────────────────────────────────────────

def resolve_pred_dir(model_name, use_relative, root_dir):
    """
    Return the depth-prediction directory for the given model and depth type.

    Parameters
    ----------
    model_name   : str   – model shortcode (DA, MD, ZD, UD, UN, AB)
    use_relative : bool  – True → relative depth, False → metric depth
    root_dir     : str   – root directory containing rltv_depths/ and metric_depths/

    Raises
    ------
    ValueError if model_name is not recognised.
    """
    if model_name == "MD":
        return os.path.join(root_dir, "rltv_depths", "MIDAS")
    if model_name == "DA" and use_relative:
        return os.path.join(root_dir, "rltv_depths", "DA")
    if model_name == "DA" and not use_relative:
        return os.path.join(root_dir, "metric_depths", "DA")
    if model_name == "ZD":
        return os.path.join(root_dir, "metric_depths", "ZOEDEPTH")
    if model_name == "UD":
        return os.path.join(root_dir, "metric_depths", "UNIDEPTH")
    if model_name == "UN":
        return os.path.join(root_dir, "metric_depths", "UNIK3D")
    if model_name == "AB":
        return os.path.join(root_dir, "metric_depths", "ADABINS")
    raise ValueError(f"Unsupported model_name: '{model_name}'")


# ── Video ↔ depth file pairing ────────────────────────────────────────────────

def _normalized_stem(path):
    """
    Strip all known video/depth extensions from a filename to get a clean stem
    for matching (e.g. 'vid.AVI.npy' → 'vid').
    """
    exts_to_strip = {".npy", ".npz", ".avi", ".mp4", ".mov", ".mkv"}
    base = os.path.basename(path).lower()
    while True:
        stem, ext = os.path.splitext(base)
        if ext in exts_to_strip and stem:
            base = stem
        else:
            break
    return base


def pair_video_and_depth_files(camera_dir, depth_dir,
                               video_exts=VIDEO_EXTS, depth_exts=DEPTH_EXTS):
    """
    Match video files in camera_dir to depth files in depth_dir by stem name.

    Parameters
    ----------
    camera_dir : str – directory containing raw video files
    depth_dir  : str – directory containing .npy depth predictions

    Returns
    -------
    paired_files  : list of (video_path, depth_path) tuples
    missing_depth : list of video basenames with no matching depth file
    extra_depth   : list of depth basenames with no matching video
    """
    video_files = sorted(
        os.path.join(camera_dir, f)
        for f in os.listdir(camera_dir)
        if f.lower().endswith(video_exts)
    )
    depth_files = sorted(
        os.path.join(depth_dir, f)
        for f in os.listdir(depth_dir)
        if f.lower().endswith(depth_exts)
    )

    depth_by_stem = {_normalized_stem(p): p for p in depth_files}

    paired_files      = []
    missing_depth     = []
    used_depth_stems  = set()

    for vid_file in video_files:
        stem       = _normalized_stem(vid_file)
        depth_file = depth_by_stem.get(stem)
        if depth_file is None:
            missing_depth.append(os.path.basename(vid_file))
            paired_files.append((vid_file, None))
        else:
            paired_files.append((vid_file, depth_file))
            used_depth_stems.add(stem)

    extra_depth = [
        os.path.basename(p)
        for p in depth_files
        if _normalized_stem(p) not in used_depth_stems
    ]

    return paired_files, missing_depth, extra_depth


# ── Site-mean depth loading ───────────────────────────────────────────────────

def load_site_mean_pred(site, pred_dir):
    """
    Load the robust-mean disparity map for a site (used in background fitting).

    Looks for files matching  <site>_robust_mean_stage_*.npy  and loads the
    first one (lexicographic order).

    Returns
    -------
    site_mean_pred      : H×W array or None
    site_mean_pred_norm : H×W array normalised to [0, 1], or None
    site_mean_path      : path that was loaded, or None
    """
    site_mean_dir   = os.path.join(pred_dir, "_SITE_MEANS_", "_means_")
    site_mean_paths = sorted(
        glob(os.path.join(site_mean_dir, f"{site}_robust_mean_stage_*.npy"))
    )
    if not site_mean_paths:
        return None, None, None

    site_mean_path = site_mean_paths[0]
    site_mean_pred = np.load(site_mean_path)

    site_mean_max = np.nanmax(site_mean_pred)
    if not np.isfinite(site_mean_max) or site_mean_max <= 0:
        return site_mean_pred, None, site_mean_path

    site_mean_pred_norm = site_mean_pred / site_mean_max
    return site_mean_pred, site_mean_pred_norm, site_mean_path


# ── Frame and mask loading ────────────────────────────────────────────────────

def read_frame_safe(raw_video, site, vid_name, frame_idx):
    """
    Read the next frame from an open VideoCapture.

    Returns None and prints a warning if the read fails, allowing the caller
    to break the frame loop cleanly instead of processing a corrupt frame.
    """
    ret, frame = raw_video.read()
    if not ret or frame is None:
        print(f"[WARN] Frame read failed: {site}/{vid_name} frame {frame_idx}. "
              f"Stopping this video early.")
        return None
    return frame


def load_animal_mask(frame_key, mask_dir_path, frame, animal_bboxes_frame,
                     mask_loader_fn=None):
    """
    Load a pre-saved animal segmentation mask, or generate one on the fly.

    Pre-saved masks (PNG, 0 = background, 255 = animal) are preferred because
    they are fast and deterministic.  If no saved mask exists and mask_loader_fn
    is provided, it is called to generate a mask (e.g. via SAM).

    Parameters
    ----------
    frame_key         : str  – frame identifier, e.g. 'frame_000042'
    mask_dir_path     : str  – directory containing per-frame mask PNGs
    frame             : H×W×3 RGB array  – used only if generating a mask live
    animal_bboxes_frame : list  – animal bboxes for this frame
    mask_loader_fn    : callable or None  – signature: (frame, bboxes) → mask

    Returns
    -------
    mask : H×W uint8 array (0/1), or None if no mask is available
    """
    mask_path = os.path.join(mask_dir_path, f"{frame_key}.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return (mask > 0).astype(np.uint8)
        return None

    if mask_loader_fn is None:
        return None
    return mask_loader_fn(frame, animal_bboxes_frame)


# ── Depth map utilities ───────────────────────────────────────────────────────

def normalize_depth_map(depth, eps=1e-8):
    """
    Normalise a depth map to [0, 1] by dividing by its maximum finite value.

    Returns a zero map if the input has no finite positive values.
    """
    depth     = np.asarray(depth, dtype=np.float32)
    depth_max = np.nanmax(depth)
    if not np.isfinite(depth_max) or depth_max <= eps:
        return np.zeros_like(depth, dtype=np.float32)
    return depth / depth_max


def projected_depth_from_distance(distance, points, u_map, v_map, focal_length):
    """
    Convert a ground-truth radial distance to projected depth Z for each point.

    The camera projects a 3-D point at radial distance D to depth Z via:

        Z = D / sqrt( u²/f² + v²/f² + 1 )

    where (u, v) are the pixel offsets from the principal point and f is the
    focal length.

    Parameters
    ----------
    distance    : float – radial distance to the target (metres)
    points      : list of (px, py) pixel coordinates
    u_map       : H×W array – x-offset from principal point (pre-computed)
    v_map       : H×W array – y-offset from principal point (pre-computed)
    focal_length: float – camera focal length in pixels

    Returns
    -------
    z_values : list of float – projected depth for each point
    """
    fl_sq = float(focal_length) ** 2
    return [
        distance / np.sqrt(u_map[py, px] ** 2 / fl_sq
                           + v_map[py, px] ** 2 / fl_sq + 1.0)
        for (px, py) in points
    ]


# ── Animal detection ──────────────────────────────────────────────────────────

def detect_animals(detection_model, frame_rgb):
    """Run MegaDetector on a single RGB frame and return a list of bbox dicts."""
    detection_result = detection_model.single_image_detection(frame_rgb)
    detections = detection_result.get("detections", None)
    if detections is None:
        return []

    xyxy_arr = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
    if xyxy_arr.size == 0:
        return []

    confidence_arr = np.asarray(getattr(detections, "confidence", []), dtype=np.float32)
    if confidence_arr.shape[0] != xyxy_arr.shape[0]:
        confidence_arr = np.ones(xyxy_arr.shape[0], dtype=np.float32)

    return [
        {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(conf),
        }
        for (x1, y1, x2, y2), conf in zip(xyxy_arr, confidence_arr)
    ]


def init_detection_model(device):
    """Load MegaDetector V6. Call lazily — only when bbox JSON is not available."""
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    logging.getLogger("ultralytics").propagate = False
    from ultralytics.utils import LOGGER as _ul_logger
    _ul_logger.setLevel(logging.ERROR)
    from PytorchWildlife.models import detection as pw_detection
    return pw_detection.MegaDetectorV6(device=device, pretrained=True, version="MDV6-yolov10-e")
