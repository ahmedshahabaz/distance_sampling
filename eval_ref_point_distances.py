"""
eval_ref_point_distances.py – Evaluate reference-point depth predictions across all sites
and save the aggregated results to a JSON file.

Usage
-----
    python eval_ref_point_distances.py <model> --relative | --metric [--calb]
                                       [--root-dir DIR] [--data-dir DIR]
                                       [--mask-dir DIR] [--animal-box-dir DIR]
                                       [--bg-fit] [--bg-fit-method {least_squares,ransac}]

    model      : DA | MD | ZD | UD | UN | AB
    --relative : use relative depth (calibration always applied)
    --metric   : use metric depth
    --calb     : apply disparity calibration (metric depth only)

Path overrides (all have sensible defaults):
    --root-dir        : root directory for depth predictions
    --data-dir        : directory containing video files and annotations
    --mask-dir        : directory of per-frame animal segmentation masks
    --animal-box-dir  : directory of animal bounding-box JSON files

Output
------
    JSON file written to the model's prediction directory:
        ref_point_stats.json      (uncalibrated run)
        ref_point_stats_calb.json (calibrated run)

    The JSON is organised as:
        site -> video index -> {
            "file_name": ...,
            "circles": [...]
        }

    Each entry in "circles" corresponds to one reference-point circle and stores:
        - the reference-point ID and GT distance
        - the projected GT depth used for evaluation ("Z")
        - median prediction summaries (raw disparity / normalised disparity /
          calibrated depth variants)
        - an "errors" dict with evaluation metrics for each calibration method
          ("fxd", "mean", "median", "all", "outliers_rmv"), including
          diff_err, abs_err, abs_rel, sq_rel, rmse, delta1, delta2, and delta3
"""

import argparse
import json
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from infer_utils import get_depth, initialize_model
from utils.calb_utils import (
    align_disparity_scale_shift,
    fit_background_to_site_mean,
    get_circle_info,
    get_combined_sam_mask,
    get_depth_on_circle,
    get_site_patch_mask,
    split_train_test,
    update_patch_stats,
    update_test_stats,
    update_vid_stats,
)
from utils.pipeline_utils import (
    load_animal_mask,
    detect_animals,
    init_detection_model,
    load_site_mean_pred,
    normalize_depth_map,
    pair_video_and_depth_files,
    projected_depth_from_distance,
    read_frame_safe,
    resolve_pred_dir,
)
from utils.script_utils import (
    add_dataset_path_args,
    add_depth_args,
    convert_numpy_types,
    print_run_banner,
    validate_model_depth_type,
)


# ── Camera / annotation constants ─────────────────────────────────────────────

ANNOT_HEIGHT      = 4500 # Original annotation canvas height (pixels)
ANNOT_WIDTH       = 8000 # Original annotation canvas width  (pixels)
BASE_FRAME_WIDTH  = 2688 # Camera frame width  (pixels)
BASE_FRAME_HEIGHT = 1520 # Camera frame height (pixels)
CROP_BOTTOM       = 80   # Pixels to crop from the bottom of each frame
FOCAL_LENGTH      = 2955 # Camera focal length (pixels)

# ── Sites to process ──────────────────────────────────────────────────────────

# SITES = [
#     "31KL", "33KL", "33LD", "36KL", "37KL", "37PS", "38LD", "39LD",
#     "40KL", "40PS", "41KL", "41PS", "44KL", "46PS", "48KL", "48PS",
#     "49PS", "55LS", "55PS", "57LS", "57PS", "59PS", "62LN", "62PS",
#     "63LN", "71LN", "80PS", "85LN",
# ]

SITES = ['31KL', '33LD', '38LD', '40PS', '48PS', '55PS', '57PS', '63LN']

def _init_frame_stats(num_patches):
    """
    Initialise per-patch accumulators for one video.

    Returns a dict keyed by patch index; each entry holds lists that are
    filled frame-by-frame and later summarised in update_vid_stats().
    """
    return {
        i: {
            "pred_disp":            [],
            "pred_disp_norm":       [],
            "calb_depth_fxd":       [],
            "calb_depth_mean":      [],
            "calb_depth_median":    [],
            "calb_depth_all":       [],
            "calb_depth_outliers_rmv": [],
            "calb_depth_smooth":    [],   # reserved – not yet populated
            "ln_clr":               None,
            "GT":                   None,
            "Z":                    None,
            "_end_points_":         None,
        }
        for i in range(num_patches)
    }


def main():
    # ── CLI parsing ───────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Evaluate reference-point depth predictions across all sites."
    )
    add_depth_args(parser)
    add_dataset_path_args(parser)
    parser.add_argument(
        "--bg-fit",
        action="store_true",
        default=False,
        help="Align each frame's background disparity to the site mean before calibration (DA only).",
    )
    parser.add_argument(
        "--bg-fit-method",
        choices=["least_squares", "ransac"],
        default="least_squares",
        help="Background alignment method (default: least_squares).",
    )
    args = parser.parse_args()

    model_name   = args.model
    use_relative = args.relative
    use_calb     = True if use_relative else args.calb
    validate_model_depth_type(parser, model_name, use_relative)

    use_background_fit    = args.bg_fit
    background_fit_method = args.bg_fit_method

    root_dir       = args.root_dir
    data_dir       = args.data_dir
    mask_dir       = args.mask_dir
    animal_box_dir = args.animal_box_dir

    pred_dir   = resolve_pred_dir(model_name, use_relative, root_dir)
    output_dir = pred_dir
    print_run_banner(model_name, use_relative, use_calb, output_dir)

    # ── Load calibration circle annotations ───────────────────────────────────
    with open(os.path.join(data_dir, "reference_point_distance.json"), "r") as f:
        ref_pts_data = json.load(f)

    # ── Pre-compute frame geometry ────────────────────────────────────────────
    cropped_height = BASE_FRAME_HEIGHT - CROP_BOTTOM
    frame_shape    = (cropped_height, BASE_FRAME_WIDTH)

    # u_map / v_map: per-pixel offset from the principal point (used for
    # the radial-distance projection  D = Z * sqrt(u²+v²/f² + 1)).
    x_map, y_map = np.meshgrid(np.arange(BASE_FRAME_WIDTH), np.arange(cropped_height))
    u_map = x_map - BASE_FRAME_WIDTH / 2
    v_map = y_map - BASE_FRAME_HEIGHT / 2

    # ── Main loop: site → video → frame ───────────────────────────────────────
    vid_stats = {}

    depth_model_dict = initialize_model()
    depth_model = depth_model_dict["MODEL"].to(depth_model_dict["DEVICE"]).eval()
    device = depth_model_dict["DEVICE"]
    depth_args = depth_model_dict["ARGS"]
    depth_transform = depth_model_dict.get("TRANSFORM", None)

    _detection_model = None  # lazily initialised if bbox JSON is missing

    for site_idx, site in enumerate(SITES):
        depth_dir = os.path.join(output_dir, site)

        # Load the site-mean disparity map (DA model only) for background fitting.
        site_mean_pred_norm = None
        if model_name == "DA" and use_background_fit:
            _, site_mean_pred_norm, _ = load_site_mean_pred(site, pred_dir)
            if site_mean_pred_norm is None:
                site_mean_dir = os.path.join(pred_dir, "_SITE_MEANS_", "_means_")
                print(f"[WARN] No site mean found for {site} in {site_mean_dir}")

        site_ref_pts = ref_pts_data[site]
        camera_dir = os.path.join(data_dir, site)
        if not os.path.isdir(depth_dir):
            os.makedirs(depth_dir, exist_ok=True)
        paired_files, missing_depth, extra_depth = pair_video_and_depth_files(camera_dir, depth_dir)

        vid_stats[site] = {}
        print(f"Processing {site}: {site_idx + 1}/{len(SITES)}")
        pbar = tqdm(total=len(paired_files))

        # Scale factors: annotation canvas → camera frame.
        scale_x = BASE_FRAME_WIDTH  / ANNOT_WIDTH
        scale_y = BASE_FRAME_HEIGHT / ANNOT_HEIGHT
        scale_r = (scale_x + scale_y) / 2

        # Mask covering all calibration patch circles (used to exclude them
        # from the background-alignment fit).
        patch_mask_site = get_site_patch_mask(site_ref_pts, scale_x, scale_y, scale_r, frame_shape)

        for vid_idx, (vid_file, depth_file) in enumerate(paired_files):
            vid_file_name = os.path.basename(vid_file)
            vid_stats[site][vid_idx] = {"file_name": vid_file_name, "circles": []}

            try:
                with open(os.path.join(animal_box_dir, site,
                                       f"{vid_file_name[:-4]}.json"), "r") as f:
                    animal_bboxes = json.load(f)
            except (FileNotFoundError, OSError):
                animal_bboxes = None

            raw_video         = cv2.VideoCapture(vid_file)
            total_frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
            all_frame_preds   = np.load(depth_file) if depth_file is not None else None
            frame_stats       = _init_frame_stats(len(site_ref_pts))
            mask_dir_path     = os.path.join(mask_dir, site, vid_file_name[:-4])

            for frame_idx in range(total_frame_count):
                frame = read_frame_safe(raw_video, site, vid_file_name, frame_idx)
                if frame is None:
                    break

                cropped_bgr = (frame[:-CROP_BOTTOM, :, :]if frame.shape[0] > CROP_BOTTOM else frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cropped_frame = (frame[:-CROP_BOTTOM, :, :]if frame.shape[0] > CROP_BOTTOM else frame)

                if all_frame_preds is not None:
                    # Resize the stored depth prediction to the cropped frame size.
                    raw_pred   = all_frame_preds[frame_idx]
                    raw_pred_t = torch.from_numpy(raw_pred).unsqueeze(0)
                    pred = F.interpolate(
                        raw_pred_t[:, None],
                        (cropped_height, BASE_FRAME_WIDTH),
                        mode="bilinear",
                        align_corners=True,
                    )[0, 0].numpy()
                else:
                    pred = get_depth(cropped_bgr, depth_model, device, depth_args, depth_transform)
                pred_norm = normalize_depth_map(pred)

                # Load or generate the animal segmentation mask for this frame.
                frame_key  = f"frame_{frame_idx:06d}"
                if animal_bboxes is not None:
                    animal_bboxes_frame = animal_bboxes.get(frame_key, [])
                else:
                    if _detection_model is None:
                        _detection_model = init_detection_model(device)
                    animal_bboxes_frame = detect_animals(_detection_model, cropped_frame)
                animal_mask_frame   = load_animal_mask(
                    frame_key, mask_dir_path, cropped_frame,
                    animal_bboxes_frame, mask_loader_fn=get_combined_sam_mask,
                )

                # Align this frame's background disparity to the site mean
                # (DA model only) to correct for per-frame exposure shifts.
                if (model_name == "DA"
                        and use_background_fit
                        and site_mean_pred_norm is not None):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        pred_norm, _, _ = fit_background_to_site_mean(
                            pred_norm, animal_bboxes_frame, animal_mask_frame,
                            patch_mask_site, frame_shape, site_mean_pred_norm,
                            method=background_fit_method,
                        )

                # ── Gather training points from all calibration circles ────
                # We pool predictions and GT depths across all visible circles
                # in this frame to fit a single per-frame (scale, shift).
                gt_depths_mean,   gt_depths_median   = [], []
                gt_depths_all,    gt_depths_no_outliers = [], []
                pred_disps_mean, pred_disps_median = [], []
                pred_disps_all,  pred_disps_no_outliers = [], []

                # Store each patch's test points so we can evaluate them
                # after fitting the per-frame calibration.
                patch_test_points = {i: [] for i in range(len(site_ref_pts))}

                for ref_pts_idx, ref_point in enumerate(site_ref_pts):
                    x, y, r, distance, unoccluded_pts = get_circle_info(
                        ref_point, animal_bboxes_frame,
                        scale_x, scale_y, scale_r,
                        frame_shape, animal_mask_frame,
                    )

                    # Skip circles that go out of frame or are fully occluded.
                    if (x - r < 0 or y - r < 0
                            or x + r >= BASE_FRAME_WIDTH
                            or y + r >= cropped_height):
                        continue
                    if not unoccluded_pts:
                        continue

                    train_pts, test_pts = split_train_test(unoccluded_pts)
                    patch_test_points[ref_pts_idx] = test_pts

                    pred_train = get_depth_on_circle(pred_norm, train_pts)
                    zs_train   = projected_depth_from_distance(
                        distance, train_pts, u_map, v_map, FOCAL_LENGTH,)

                    update_patch_stats(
                        pred_train, zs_train,
                        gt_depths_mean,  gt_depths_median,
                        gt_depths_all,   gt_depths_no_outliers,
                        pred_disps_mean, pred_disps_median,
                        pred_disps_all,  pred_disps_no_outliers,)

                if not pred_disps_mean:
                    continue  # No visible calibration circles this frame.

                # ── Fit per-frame disparity calibration ───────────────────
                # We fit four variants that differ in which training points
                # are pooled (mean/median summary vs all vs outlier-filtered).
                calib_params = {
                    "fxd": (4.0, 25.0),   # fixed fallback (not data-driven)
                    "mean": align_disparity_scale_shift(
                        np.array(pred_disps_mean),
                        np.array(gt_depths_mean),
                    ),
                    "median": align_disparity_scale_shift(
                        np.array(pred_disps_median),
                        np.array(gt_depths_median),
                    ),
                    "all": align_disparity_scale_shift(
                        np.array(pred_disps_all),
                        np.array(gt_depths_all),
                    ),
                    "outliers_rmv": align_disparity_scale_shift(
                        np.array(pred_disps_no_outliers),
                        np.array(gt_depths_no_outliers),
                    ),
                }

                # ── Evaluate held-out test points for each patch ───────────
                for ref_pts_idx, ref_point in enumerate(site_ref_pts):
                    x        = int(ref_point["x"] * scale_x)
                    y        = int(ref_point["y"] * scale_y)
                    distance = ref_point["distance"]

                    if not (0 <= x < BASE_FRAME_WIDTH and 0 <= y < cropped_height):
                        continue

                    test_pts = patch_test_points[ref_pts_idx]
                    if not test_pts:
                        continue

                    pred_test_norm = get_depth_on_circle(pred_norm, test_pts)
                    pred_test      = get_depth_on_circle(pred, test_pts)
                    zs_test        = projected_depth_from_distance(
                        distance, test_pts, u_map, v_map, FOCAL_LENGTH,)

                    update_test_stats(frame_stats, ref_pts_idx,pred_test, pred_test_norm,
                        distance, zs_test,calib_params, use_relative, use_calb,)

                # Summarise this video's frame-level stats.
                update_vid_stats(vid_stats, site, vid_idx, frame_stats)

            raw_video.release()
            pbar.update(1)

        pbar.close()

    # ── Write results ─────────────────────────────────────────────────────────
    stats_filename = ("ref_point_stats_calb.json" if use_calb
                      else "ref_point_stats.json")
    stats_path = os.path.join(output_dir, stats_filename)
    with open(stats_path, "w") as f:
        json.dump(vid_stats, f, indent=4, default=convert_numpy_types)

    print(os.path.basename(stats_path))


if __name__ == "__main__":
    main()
