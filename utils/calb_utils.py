"""
calb_utils.py – Calibration and annotation utilities.

Sections
--------
1. Circle / patch sampling utilities
2. Background alignment (least-squares and RANSAC)
3. Disparity calibration and error metrics
4. Per-frame / per-video stats aggregation
5. Frame annotation (distance labels)
6. SAM segmentation-mask utilities
"""

import random

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

# Optional heavy dependencies — degrade gracefully if unavailable.
try:
    import torch
    _TORCH_IMPORT_ERROR = None
except Exception as e:
    torch = None
    _TORCH_IMPORT_ERROR = e

try:
    from transformers import SamModel, SamProcessor
    _SAM_IMPORT_ERROR = None
except Exception as e:
    SamModel = None
    SamProcessor = None
    _SAM_IMPORT_ERROR = e


# ── 1. Circle / patch sampling utilities ─────────────────────────────────────

def safe_median(arr):
    return np.median(arr) if len(arr) > 0 else np.nan


def get_depth_on_circle(depth_map, circle_pts):
    """Return depth values at the given (x, y) pixel coordinates."""
    return [depth_map[cy, cx] for (cx, cy) in circle_pts]


def split_train_test(circle_pts, train_frac=0.8):
    """Randomly split circle edge points into train / test subsets."""
    if len(circle_pts) <= 2:
        return circle_pts, []
    random.shuffle(circle_pts)
    split_idx = int(len(circle_pts) * train_frac)
    return circle_pts[:split_idx], circle_pts[split_idx:]


def get_animal_mask(animal_bboxes_frame, frame_shape, conf_thresh=0.45):
    """Binary mask (H×W) that is 1 inside any high-confidence animal bbox."""
    mask = np.zeros(frame_shape, dtype=np.uint8)
    for bbox in animal_bboxes_frame:
        if bbox.get('confidence', 1.0) >= conf_thresh:
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            mask[y1:y2, x1:x2] = 1
    return mask


def get_background_mask(animal_bboxes_frame, patch_mask_site, frame_shape):
    """
    Background mask: pixels that are neither an animal nor a calibration patch.
    Used to select pixels for background-to-reference alignment.
    """
    animal_mask = get_animal_mask(animal_bboxes_frame, frame_shape, conf_thresh=0.45)
    if patch_mask_site is not None:
        return (animal_mask == 0) & (patch_mask_site == 0)
    return animal_mask == 0


def get_site_patch_mask(ref_pts, scale_x, scale_y, scale_r, frame_shape):
    """Binary mask (H×W) covering all calibration circle patches for a site."""
    mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    for ref_point in ref_pts:
        x = int(ref_point['x'] * scale_x)
        y = int(ref_point['y'] * scale_y)
        r = int(ref_point['radius'] * scale_r)
        cv2.circle(mask, (x, y), r, color=1, thickness=-1)
    return mask


def get_circle_info(ref_point, animal_bboxes_frame, scale_x, scale_y, scale_r,
                    frame_shape, animal_mask=None):
    """
    Scale an annotation circle to frame coordinates and return its unoccluded
    edge points (i.e. points not covered by any detected animal).

    Parameters
    ----------
    ref_point : dict  – annotation entry with 'x', 'y', 'radius', 'distance'
    animal_mask : 2-D uint8 array or None  – pre-computed animal mask

    Returns
    -------
    x, y, r : int  – circle centre and radius in frame pixels
    distance : float  – ground-truth distance in metres
    unoccluded_pts : list of (px, py) tuples
    """
    x = int(ref_point['x'] * scale_x)
    y = int(ref_point['y'] * scale_y)
    r = int(ref_point['radius'] * scale_r)
    distance = ref_point['distance']

    cropped_height, width = frame_shape

    if animal_mask is None or animal_mask.sum() == 0:
        animal_mask = get_animal_mask(animal_bboxes_frame, frame_shape)

    edge_pts = cv2.ellipse2Poly((x, y), (r, r), 0, 0, 360, delta=5)
    edge_pts = [tuple(pt) for pt in edge_pts
                if 0 <= pt[0] < width and 0 <= pt[1] < cropped_height]
    unoccluded_pts = [(px, py) for (px, py) in edge_pts if animal_mask[py, px] == 0]

    return x, y, r, distance, unoccluded_pts


# ── 2. Background alignment ───────────────────────────────────────────────────

def _least_squares_align_background(obs_bg, ref_bg, mask=None):
    """
    Fit a linear mapping  ref ≈ scale * obs + shift  via ordinary least squares.

    Parameters
    ----------
    obs_bg : 1-D array  – per-frame background disparity values
    ref_bg : 1-D array  – site-mean background disparity values (same shape)
    mask   : bool array or None  – restrict fit to these pixels

    Returns
    -------
    scale, shift : float
    """
    obs_bg = np.asarray(obs_bg).flatten()
    ref_bg = np.asarray(ref_bg).flatten()

    valid = mask.flatten() & np.isfinite(obs_bg) & np.isfinite(ref_bg) \
        if mask is not None else np.isfinite(obs_bg) & np.isfinite(ref_bg)

    x = obs_bg[valid].astype(np.float64, copy=False)
    y = ref_bg[valid].astype(np.float64, copy=False)

    if x.size < 2 or np.nanstd(x) < 1e-10:
        return 1.0, 0.0

    A = np.vstack([x, np.ones_like(x)]).T
    scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]

    if not np.isfinite(scale) or not np.isfinite(shift):
        return 1.0, 0.0
    return scale, shift


def _ransac_align_background(obs_bg, ref_bg, mask=None,
                              min_samples=2, residual_threshold=0.05,
                              random_state=None):
    """
    Fit a robust linear mapping  ref ≈ scale * obs + shift  via RANSAC,
    ignoring outlier background pixels.

    Parameters
    ----------
    obs_bg : 1-D array  – per-frame background disparity values
    ref_bg : 1-D array  – site-mean background disparity values (same shape)
    mask   : bool array or None  – restrict fit to these pixels

    Returns
    -------
    scale, shift : float
    """
    obs_bg = np.asarray(obs_bg).flatten()
    ref_bg = np.asarray(ref_bg).flatten()

    valid = mask.flatten() & np.isfinite(obs_bg) & np.isfinite(ref_bg) \
        if mask is not None else np.isfinite(obs_bg) & np.isfinite(ref_bg)

    x = obs_bg[valid].astype(np.float64, copy=False).reshape(-1, 1)
    y = ref_bg[valid].astype(np.float64, copy=False).reshape(-1, 1)

    if x.shape[0] < 2 or np.nanstd(x) < 1e-10:
        return 1.0, 0.0

    ransac = RANSACRegressor(
        LinearRegression(),
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        random_state=random_state,
    )
    ransac.fit(x, y)
    scale = ransac.estimator_.coef_[0, 0]
    shift = ransac.estimator_.intercept_[0]

    if not np.isfinite(scale) or not np.isfinite(shift):
        return 1.0, 0.0
    return scale, shift


def fit_background_to_site_mean(
    pred_norm,
    animal_bboxes_frame,
    all_animal_mask_frame,
    patch_mask_site,
    frame_shape,
    site_mean_pred_norm,
    method="ransac",
    residual_threshold=0.05,
    random_state=0,
):
    """
    Align a per-frame normalised disparity map to the site-mean background.

    The alignment is a per-frame linear scale+shift fitted on background pixels
    (pixels not covered by animals or calibration patches).

    Returns
    -------
    pred_norm_aligned : np.ndarray  – aligned disparity map
    scale, shift      : float       – fitted parameters
    """
    if site_mean_pred_norm is None:
        return pred_norm, 1.0, 0.0

    ref_pred = np.asarray(site_mean_pred_norm)
    if ref_pred.shape != pred_norm.shape:
        ref_pred = cv2.resize(
            ref_pred.astype(np.float32),
            (pred_norm.shape[1], pred_norm.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Build background mask (exclude animals and calibration patches).
    if all_animal_mask_frame is not None:
        animal_mask = np.asarray(all_animal_mask_frame)
        if animal_mask.shape != pred_norm.shape:
            animal_mask = cv2.resize(
                animal_mask.astype(np.uint8),
                (pred_norm.shape[1], pred_norm.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        bkgrnd_mask = ((animal_mask > 0) == 0)  # True where no animal
    else:
        bkgrnd_mask = get_background_mask(animal_bboxes_frame, patch_mask_site, frame_shape)
        if bkgrnd_mask.shape != pred_norm.shape:
            bkgrnd_mask = cv2.resize(
                bkgrnd_mask.astype(np.uint8),
                (pred_norm.shape[1], pred_norm.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

    obs_bg = np.asarray(pred_norm[bkgrnd_mask]).reshape(-1)
    ref_bg = np.asarray(ref_pred[bkgrnd_mask]).reshape(-1)

    # Guard against shape mismatches after masking.
    if obs_bg.size != ref_bg.size:
        keep = min(obs_bg.size, ref_bg.size)
        if keep < 2:
            return pred_norm, 1.0, 0.0
        obs_bg = obs_bg[:keep]
        ref_bg = ref_bg[:keep]

    # Keep only finite, non-extreme values.
    valid = np.isfinite(obs_bg) & np.isfinite(ref_bg)
    if np.sum(valid) < 2:
        return pred_norm, 1.0, 0.0
    obs = obs_bg[valid].astype(np.float64)
    ref = ref_bg[valid].astype(np.float64)

    finite_range = (np.abs(obs) < 1e6) & (np.abs(ref) < 1e6)
    if np.sum(finite_range) < 2:
        return pred_norm, 1.0, 0.0
    obs = obs[finite_range]
    ref = ref[finite_range]

    if np.nanstd(obs) < 1e-10:
        return pred_norm, 1.0, 0.0

    # Fit the alignment.
    method = method.lower()
    if method in ("ransac", "robust"):
        try:
            scale, shift = _ransac_align_background(
                obs, ref,
                residual_threshold=residual_threshold,
                random_state=random_state,
            )
        except Exception:
            scale, shift = _least_squares_align_background(obs, ref)
    elif method in ("least_squares", "least", "ls"):
        scale, shift = _least_squares_align_background(obs, ref)
    else:
        raise ValueError(f"Unsupported background fit method: '{method}'. "
                         f"Choose 'ransac' or 'least_squares'.")

    if not np.isfinite(scale) or not np.isfinite(shift):
        return pred_norm, 1.0, 0.0

    return pred_norm * scale + shift, scale, shift


# ── 3. Disparity calibration and error metrics ───────────────────────────────

def align_disparity_scale_shift(pred_disp, gt_depth):
    """
    Fit a linear mapping from predicted disparity to ground-truth inverse depth.

    The model predicts *disparity* d̂ ≈ 1/Z.  We find (scale, shift) such that:

        1/Z  ≈  scale * d̂  +  shift

    Parameters
    ----------
    pred_disp : 1-D array  – predicted disparity values for calibration points
    gt_depth  : 1-D array  – ground-truth depth (Z) in metres

    Returns
    -------
    scale, shift : float
    """
    gt_disp = 1.0 / gt_depth
    A = np.vstack([pred_disp, np.ones_like(pred_disp)]).T
    scale, shift = np.linalg.lstsq(A, gt_disp, rcond=None)[0]
    return scale, shift


def compute_metrics(pred, gt_depth):
    """
    Compute standard depth-estimation error metrics against a scalar GT depth.

    Parameters
    ----------
    pred     : array-like  – predicted depth values (metres)
    gt_depth : float       – ground-truth depth (metres)

    Returns
    -------
    diff, abs_err, abs_rel, sq_rel, rmse, delta1, delta2, delta3 : float
        Returns a list of NaN for each metric if inputs are invalid.
    """
    pred = np.asarray(pred).ravel()
    pred = pred[np.isfinite(pred)]
    if pred.size == 0 or not np.isfinite(gt_depth) or gt_depth <= 0:
        return [np.nan] * 8

    diff     = pred - gt_depth
    abs_err  = np.mean(np.abs(diff))
    abs_rel  = np.mean(np.abs(diff) / gt_depth)
    sq_rel   = np.mean((diff ** 2) / gt_depth)
    rmse     = np.sqrt(np.mean(diff ** 2))

    ratio  = np.maximum(pred / gt_depth, gt_depth / pred)
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25 ** 2)
    delta3 = np.mean(ratio < 1.25 ** 3)

    return np.mean(diff), abs_err, abs_rel, sq_rel, rmse, delta1, delta2, delta3


# ── 4. Per-frame / per-video stats aggregation ───────────────────────────────

def update_patch_stats(
    pred_train, zs_train,
    gt_depth_frame_mean, gt_depth_frame_median,
    gt_depth_frame_all, gt_depth_frame_outliers_rmv,
    pred_frame_mean, pred_frame_median,
    pred_frame_all, pred_frame_all_outliers_rmv,
):
    """
    Accumulate per-patch predicted and GT depth statistics into frame-level lists.

    For each circle patch (train split), we compute mean/median summary values
    and a 1-sigma-filtered 'outliers removed' subset, then append them to the
    running frame-level accumulators.

    Parameters
    ----------
    pred_train : list  – disparity values on the circle's training points
    zs_train   : list  – projected GT depths for those same points
    *_mean, *_median, *_all, *_outliers_rmv : lists  – accumulators to update
    """
    pred_mean   = np.mean(pred_train)
    pred_median = np.median(pred_train)

    # 1-sigma filter: keep predictions within one std of the mean.
    std = np.std(pred_train)
    lo, hi = pred_mean - std, pred_mean + std
    inlier_mask = [lo <= d <= hi for d in pred_train]

    gt_depth_frame_mean.append(np.mean(zs_train))
    gt_depth_frame_median.append(np.median(zs_train))
    gt_depth_frame_all.extend(zs_train)
    gt_depth_frame_outliers_rmv.extend(
        [z for z, keep in zip(zs_train, inlier_mask) if keep]
    )

    pred_frame_mean.append(pred_mean)
    pred_frame_median.append(pred_median)
    pred_frame_all.extend(pred_train)
    pred_frame_all_outliers_rmv.extend(
        [d for d, keep in zip(pred_train, inlier_mask) if keep])


def update_test_stats(
    frame_stats,
    patch_idx,
    pred_test,
    pred_test_norm,
    distance,
    zs_test,
    calib_params,
    use_relative=False,
    use_calb=False,
):
    """
    Append calibrated depth predictions for a patch's test points to frame_stats.

    Parameters
    ----------
    frame_stats   : dict  – per-patch accumulator (keyed by patch index)
    patch_idx     : int   – index of the current calibration circle
    pred_test     : list  – raw (metric-scale) disparity on test points
    pred_test_norm: list  – normalised disparity on test points
    distance      : float – GT distance for this patch (metres)
    zs_test       : list  – projected GT depths for test points
    calib_params  : dict  – mapping from method name → (scale, shift)
    use_relative  : bool  – use normalised disparity as calibration input
    use_calb      : bool  – same effect as use_relative for input selection
    """
    frame_stats[patch_idx]['pred_disp'].append(np.median(pred_test))
    frame_stats[patch_idx]['pred_disp_norm'].append(np.median(pred_test_norm))
    frame_stats[patch_idx]['GT'] = distance
    frame_stats[patch_idx]['Z'] = np.median(zs_test)

    # Choose which prediction to calibrate (normalised vs raw).
    base = pred_test_norm if (use_relative or use_calb) else pred_test

    for method, (scale, shift) in calib_params.items():
        disparity = np.array(base) * scale + shift
        # Convert disparity → depth, guarding against near-zero disparity.
        disp_safe = np.where(np.abs(disparity) < 1e-7, np.nan, disparity)
        inv_disp = 1.0 / disp_safe
        valid_depths = inv_disp[np.isfinite(inv_disp)]
        calb_depth = np.nan if valid_depths.size == 0 else np.nanmean(valid_depths)
        frame_stats[patch_idx][f"calb_depth_{method}"].append(calb_depth)


def update_vid_stats(vid_stats, site, vid_idx, frame_stats):
    """
    Summarise per-patch frame_stats into vid_stats after processing all frames.

    For each calibration circle, computes error metrics for every calibration
    method and appends an entry to vid_stats[site][vid_idx]['circles'].

    Parameters
    ----------
    vid_stats   : dict  – top-level output accumulator (site → video → circles)
    site        : str   – site identifier
    vid_idx     : int   – video index within the site
    frame_stats : dict  – per-patch per-frame accumulator (patch_idx → data)
    """
    for patch_idx, patch_data in frame_stats.items():
        gt       = patch_data['GT']
        gt_depth = patch_data['Z']  # projected depth from GT distance

        calib_types = {
            "fxd":          patch_data['calb_depth_fxd'],
            "mean":         patch_data['calb_depth_mean'],
            "median":       patch_data['calb_depth_median'],
            "all":          patch_data['calb_depth_all'],
            "outliers_rmv": patch_data['calb_depth_outliers_rmv'],
            # "smooth": patch_data.get('calb_depth_smooth', []),  # not yet used
        }

        errors = {}
        for name, calb_preds in calib_types.items():
            preds = np.array(calb_preds)
            diff_err, abs_err, abs_rel, sq_rel, rmse, delta1, delta2, delta3 = \
                compute_metrics(preds, gt_depth)
            errors[name] = {
                "diff_err": diff_err,
                "abs_err":  abs_err,
                "abs_rel":  abs_rel,
                "sq_rel":   sq_rel,
                "rmse":     rmse,
                "delta1":   delta1,
                "delta2":   delta2,
                "delta3":   delta3,
            }

        if not errors:
            continue

        vid_stats[site][vid_idx]["circles"].append({
            "circle_id":            str(patch_idx),
            "GT":                   gt,
            "Z":                    gt_depth,
            "pred_disp":            safe_median(patch_data['pred_disp']),
            "pred_disp_norm":       safe_median(patch_data['pred_disp_norm']),
            "calb_depth_fxd":       safe_median(patch_data['calb_depth_fxd']),
            "calb_depth_mean":      safe_median(patch_data['calb_depth_mean']),
            "calb_depth_median":    safe_median(patch_data['calb_depth_median']),
            "calb_depth_all":       safe_median(patch_data['calb_depth_all']),
            "calb_depth_outliers_rmv": safe_median(patch_data['calb_depth_outliers_rmv']),
            "errors":               errors,
        })


# ── 5. Frame annotation ───────────────────────────────────────────────────────

def annotate_animal_distances(
    frame,
    animal_bboxes_frame,
    calb_depth,
    frame_shape,
    focal_length,
    gt_distance=None,
    conf_thresh=0.45,
    site_number=None,
    file_name=None,
    site_mean_depth=None,
    combined_sam_mask=None,
    full_frame_shape=None,
):
    """
    Overlay predicted animal distances on a frame.

    For each detected animal bbox, extracts depth from the calibrated depth map
    (using a SAM mask if available, else falling back to the bbox region),
    projects depth to distance using the camera geometry, and draws a label.

    Parameters
    ----------
    frame              : H×W×3 RGB array
    calb_depth         : H×W float array – calibrated depth in metres
    frame_shape        : (cropped_height, width)
    focal_length       : float – camera focal length in pixels
    gt_distance        : float or None – if provided, GT label is drawn and
                         per-animal errors are returned
    combined_sam_mask  : H×W bool array or None – pre-computed animal SAM mask
    full_frame_shape   : (full_height, width) or None – if provided, uses
                         full_height for the principal point c_y so that
                         cropping from the bottom does not shift the optical axis

    Returns
    -------
    frame          : annotated frame
    errors         : dict of per-animal error stats (only if gt_distance given)
    """
    cropped_height, frame_width = frame_shape
    full_height = full_frame_shape[0] if full_frame_shape is not None else cropped_height
    c_x = frame_width / 2
    c_y = full_height / 2
    errors = {}
    frame_org = frame.copy()

    if combined_sam_mask is None and animal_bboxes_frame:
        combined_sam_mask = get_combined_sam_mask(frame, animal_bboxes_frame, conf_thresh)

    frame_rgb_mask = frame_org.copy()

    for animal_idx, bbox in enumerate(animal_bboxes_frame):
        conf = bbox.get('confidence', None)
        if conf is None or conf < conf_thresh:
            continue

        x1 = max(0, int(bbox['x1']))
        y1 = max(0, int(bbox['y1']))
        x2 = min(calb_depth.shape[1], int(bbox['x2']))
        y2 = min(calb_depth.shape[0], int(bbox['y2']))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 50)

        region = calb_depth[y1:y2, x1:x2]
        if region.size == 0:
            continue

        # Per-pixel (u, v) offsets from principal point for radial correction.
        yy, xx = np.meshgrid(np.arange(y1, y2), np.arange(x1, x2), indexing='ij')
        u = xx - c_x
        v = yy - c_y

        # Depth extraction: prefer SAM mask, fall back to raw bbox region.
        if combined_sam_mask is not None:
            region_mask = combined_sam_mask[y1:y2, x1:x2]
            masked_depths = region[region_mask]
            masked_depths = masked_depths[masked_depths > 0]
            sam_animal_depth = float(np.nanmedian(masked_depths)) \
                if masked_depths.size > 0 else np.nan
        else:
            sam_animal_depth = get_sam_mask_depth(frame, calb_depth, x1, y1, x2, y2)

        # Project box depth → distance using radial factor  D = Z * sqrt(u²+v²/f² + 1).
        fl_sq = float(focal_length) ** 2
        radial_factor = np.sqrt((u ** 2 + v ** 2) / fl_sq + 1.0)
        region_safe = np.where(np.isfinite(region), region, np.nan)
        finite_vals = np.isfinite(region_safe)
        if finite_vals.any():
            box_animal_distance = round(float(np.nanmean(region_safe * radial_factor)), 1)
        else:
            box_animal_distance = np.nan

        if np.isfinite(sam_animal_depth):
            sam_animal_dist = round(float(sam_animal_depth * np.nanmedian(radial_factor)), 1)
        else:
            sam_animal_dist = np.nan
        
        if gt_distance is not None:
            diff    = sam_animal_dist - gt_distance
            abs_err = np.abs(diff)
            abs_rel = np.abs(diff) / gt_distance
            errors[animal_idx] = {
                "pred_box":  box_animal_distance,
                "pred_mask": sam_animal_dist,
                "diff_err":  diff,
                "abs_err":   abs_err,
                "abs_rel":   abs_rel,
            }

        label_box  = f"PRED:{box_animal_distance:.1f}m"
        label_mask = f"PRED_SAM:{sam_animal_dist:.1f}m"

        frame_rgb_mask = visualize_sam_mask_on_frame(frame_org, combined_sam_mask)
        text_x = x1 + 20
        text_y = max(y1 - 55, 10)
        cv2.putText(frame, label_box, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5, cv2.LINE_AA)
        
        cv2.putText(frame_rgb_mask, label_mask, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5, cv2.LINE_AA)

        if gt_distance is not None:
            gt_label = f"GT:{gt_distance:.1f}m"
            cv2.putText(frame, gt_label, (text_x, text_y + 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5, cv2.LINE_AA)

    if gt_distance is not None:
        return frame_rgb_mask, frame, errors
    return frame_rgb_mask


# ── 6. SAM segmentation-mask utilities ───────────────────────────────────────

# Module-level SAM model state (lazy-loaded on first use).
device        = None
sam_model     = None
sam_processor = None
_sam_warned              = False
_sam_depth_fallback_warned = False


def _ensure_sam_loaded():
    """Load the SAM model and processor once; return True if ready."""
    global device, sam_model, sam_processor, _sam_warned

    if sam_model is not None and sam_processor is not None:
        return True

    if torch is None or SamModel is None or SamProcessor is None:
        if not _sam_warned:
            missing = _TORCH_IMPORT_ERROR or _SAM_IMPORT_ERROR
            print(f"[WARN] SAM unavailable – import failed: {missing}")
            _sam_warned = True
        return False

    try:
        device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam_model     = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        return True
    except Exception as e:
        if not _sam_warned:
            print(f"[WARN] Could not load SAM model: {e}")
            _sam_warned = True
        sam_model = sam_processor = device = None
        return False


def get_combined_sam_mask(frame, animal_bboxes_frame, conf_thresh=0.45):
    """
    Run SAM for every high-confidence animal bbox and OR the masks together.

    Returns None if SAM is unavailable.
    """
    if not _ensure_sam_loaded():
        return None
    combined = np.zeros(frame.shape[:2], dtype=bool)
    for bbox in animal_bboxes_frame:
        conf = bbox.get('confidence', None)
        if conf is None or conf < conf_thresh:
            continue
        x1, y1 = int(bbox['x1']), int(bbox['y1'])
        x2, y2 = int(bbox['x2']), int(bbox['y2'])
        combined |= get_sam_mask(frame, x1, y1, x2, y2, out_shape=frame.shape[:2])
    return combined


def get_sam_mask(frame, x1, y1, x2, y2, out_shape=None):
    """
    Run SAM on a single bounding box and return a binary pixel mask.

    Parameters
    ----------
    frame    : H×W×3 RGB array
    x1 … y2 : bbox coordinates
    out_shape: (height, width) to resize the output mask to

    Returns
    -------
    mask : bool array of shape out_shape (or frame.shape[:2])
    """
    if not _ensure_sam_loaded():
        raise RuntimeError("SAM is unavailable in this environment.")

    inputs = sam_processor(
        frame, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # (num_masks, H, W)

    mask = masks[0].numpy()

    # Normalise to 2-D (SAM can return (3,H,W) or (H,W,3) or (H,W)).
    if mask.ndim == 3 and mask.shape[0] == 3:
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[-1] == 3:
        mask = mask[..., 0]
    elif mask.ndim != 2:
        raise ValueError(f"Unexpected SAM mask shape: {mask.shape}")

    if mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError(f"SAM mask has invalid shape {mask.shape}")

    target = out_shape if out_shape is not None else frame.shape[:2]
    mask_resized = cv2.resize(
        mask.astype(np.uint8),
        (target[1], target[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return mask_resized.astype(bool)


def get_sam_mask_depth(frame, depth_map, x1, y1, x2, y2):
    """
    Return the median depth inside a SAM-generated animal mask.

    Falls back to raw bbox-region depth if SAM is unavailable or fails.

    Parameters
    ----------
    frame     : H×W×3 RGB array
    depth_map : 2-D float array (may differ in shape from frame)

    Returns
    -------
    depth : float (metres), or np.nan if no valid pixels found
    """
    global _sam_depth_fallback_warned

    try:
        mask = get_sam_mask(frame, x1, y1, x2, y2, out_shape=depth_map.shape)
        masked_depth = depth_map[mask]
    except Exception as e:
        if not _sam_depth_fallback_warned:
            print(f"[WARN] SAM mask failed, falling back to bbox depth: {e}")
            _sam_depth_fallback_warned = True
        # Clamp coords to depth_map boundaries.
        x1 = max(0, int(x1));  y1 = max(0, int(y1))
        x2 = min(depth_map.shape[1], int(x2))
        y2 = min(depth_map.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            return np.nan
        masked_depth = depth_map[y1:y2, x1:x2]

    valid = masked_depth[np.isfinite(masked_depth) & (masked_depth > 0)]
    if valid.size == 0:
        return np.nan
    return round(float(np.nanmedian(valid)), 1)


def visualize_sam_mask_on_frame(frame, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlay a binary SAM mask on an RGB frame with a semi-transparent colour.

    Parameters
    ----------
    frame : H×W×3 RGB array
    mask  : 2-D bool/int array (will be resized if shapes differ)
    color : BGR tuple for the overlay colour
    alpha : opacity of the overlay (0 = transparent, 1 = opaque)

    Returns
    -------
    overlay : H×W×3 RGB array
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.any(mask, axis=-1)
    elif mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape for visualization: {mask.shape}")

    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,)

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    color_layer = np.empty_like(frame)
    color_layer[:] = color
    blended = cv2.addWeighted(frame, 1 - alpha, color_layer, alpha, 0)
    overlay = frame.copy()
    cv2.copyTo(blended, mask_u8, overlay)
    return overlay
