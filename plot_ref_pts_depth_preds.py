"""
plot_ref_pts_depth_preds.py – Annotate videos with predicted distances and generate variance plots.

Usage
-----
    python plot_ref_pts_depth_preds.py [--save-videos]

This script is fixed to the `DA` model with relative depth (`DA_rltv`).
    --save-videos / --save-vids : write annotated frames as .mp4 videos;
                    by default only variance / VMR plots are saved

For each video the script:
  1. Loads pre-computed depth predictions (.npy) when available; otherwise
     runs depth inference on the fly, and loads animal bounding boxes.
  2. For every frame, fits a per-frame disparity calibration against the
     circle-patch GT distances, then converts disparity → metric depth.
  3. Annotates animals with their predicted distance on each frame.
  4. If --save-videos is set, writes the annotated frames to an .mp4 file.
  5. Stacks normalised/calibrated depth maps and saves variance / VMR plots.

Output directories
------------------
    --out-vid-dir/<model_name>/<site>/  – annotated videos (--save-videos only)
    --out-plot-dir/<model_name>/<site>/ – variance / VMR plots (always)
"""

import argparse
import gc
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from infer_utils import get_depth, initialize_model
from utils.analysis_utils import plot_depth_variance, update_bbox_results
from utils.calb_utils import (
    align_disparity_scale_shift,
    annotate_animal_distances,
    get_circle_info,
    get_combined_sam_mask,
    get_depth_on_circle,
    visualize_sam_mask_on_frame,
)
from utils.pipeline_utils import (
    detect_animals,
    init_detection_model,
    load_animal_mask,
    load_site_mean_pred,
    normalize_depth_map,
    pair_video_and_depth_files,
    projected_depth_from_distance,
    read_frame_safe,
    resolve_pred_dir,
)
from utils.script_utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_ROOT_DIR,
    add_dataset_path_args,
    print_run_banner,
)


# ── Camera / annotation constants ─────────────────────────────────────────────

ANNOT_HEIGHT      = 4500 # Original annotation canvas height (pixels)
ANNOT_WIDTH       = 8000 # Original annotation canvas width  (pixels)
BASE_FRAME_WIDTH  = 2688 # Camera frame width  (pixels)
BASE_FRAME_HEIGHT = 1520 # Camera frame height (pixels)
CROP_BOTTOM       = 80   # Pixels to crop from the bottom of each frame
FOCAL_LENGTH      = 2955 # Camera focal length (pixels)

# ── Sites to process ──────────────────────────────────────────────────────────
# This plotting script intentionally uses a smaller site subset so the output
# stays manageable. Edit SITES below if you want a different set of sites.

# SITES = ["31KL", "33KL", "33LD", "36KL", "37KL", "37PS", "38LD", "39LD",
#     "40KL", "40PS", "41KL", "41PS", "44KL", "46PS", "48KL", "48PS",
#     "49PS", "55LS", "55PS", "57LS", "57PS", "59PS", "62LN", "62PS",
#     "63LN", "71LN", "80PS", "85LN"]

SITES = ['31KL', '33LD', '38LD', '40PS', '48PS', '55PS', '57PS', '63LN']

# ── Videos to process within the selected sites ───────────────────────────────

def main():
    # ── CLI parsing ───────────────────────────────────────────────────────────
    _ds_root = "./outputs/"
    parser = argparse.ArgumentParser(
        description=(
            "Annotate videos with predicted distances and generate variance plots "
            "for the fixed DA relative-depth workflow."
        )
    )

    parser.add_argument(
        "--root-dir",
        default=DEFAULT_ROOT_DIR,
        metavar="DIR",
        help="Root directory for depth predictions. (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        metavar="DIR",
        help="Directory containing video files and annotations. (default: %(default)s)",
    )
    add_dataset_path_args(parser)
    parser.add_argument(
        "--save-videos", "--save-vids",
        action="store_true",
        dest="save_videos",
        help="Write annotated frames as .mp4 videos (default: plots only).",
    )
    parser.add_argument(
        "--out-vid-dir",
        default=f"{_ds_root}/output_distances_vids",
        metavar="DIR",
        help="Root directory for annotated video output. (default: %(default)s)",
    )
    parser.add_argument(
        "--out-plot-dir",
        default=f"{_ds_root}/plots",
        metavar="DIR",
        help="Root directory for variance/VMR plot output. (default: %(default)s)",
    )
    args = parser.parse_args()

    model_name     = "DA"
    use_relative   = True
    use_calb       = True
    save_videos    = args.save_videos
    data_dir       = args.data_dir
    mask_dir       = args.mask_dir
    animal_box_dir = args.animal_box_dir

    pred_dir = resolve_pred_dir(model_name, use_relative, args.root_dir)

    # Use model + depth-type as the subdir key so DA_rltv and DA_metric
    # never share the same output directory.
    depth_tag    = "rltv" if use_relative else "metric"
    model_key    = f"{model_name}_{depth_tag}"
    out_vid_dir  = os.path.join(args.out_vid_dir,  model_key)
    out_plot_dir = os.path.join(args.out_plot_dir, model_key)

    print_run_banner(model_name, use_relative, use_calb, pred_dir)

    # ── Load calibration circle annotations ───────────────────────────────────
    with open(os.path.join(data_dir, "reference_point_distance.json"), "r") as f:
        ref_pts_data = json.load(f)

    # ── Pre-compute frame geometry ────────────────────────────────────────────
    cropped_height = BASE_FRAME_HEIGHT - CROP_BOTTOM
    frame_shape    = (cropped_height, BASE_FRAME_WIDTH)

    x_map, y_map = np.meshgrid(
        np.arange(BASE_FRAME_WIDTH), np.arange(cropped_height)
    )
    u_map = x_map - BASE_FRAME_WIDTH / 2
    v_map = y_map - BASE_FRAME_HEIGHT / 2

    # ── Depth model (used as fallback when no pre-computed .npy exists) ───────
    depth_model_dict  = initialize_model()
    depth_model       = depth_model_dict["MODEL"].to(depth_model_dict["DEVICE"]).eval()
    depth_device      = depth_model_dict["DEVICE"]
    depth_args        = depth_model_dict["ARGS"]
    depth_transform   = depth_model_dict.get("TRANSFORM", None)

    _detection_model = None  # lazily initialised if bbox JSON is missing

    # ── Main loop: site → video → frame ───────────────────────────────────────
    for site_idx, site in enumerate(SITES):
        depth_dir = os.path.join(pred_dir, site)

        site_ref_pts = ref_pts_data[site]
        camera_dir = os.path.join(data_dir, site)
        if not os.path.isdir(depth_dir):
            os.makedirs(depth_dir, exist_ok=True)
        paired_files, missing_depth, extra_depth = pair_video_and_depth_files(camera_dir, depth_dir)

        output_vid_path = os.path.join(out_vid_dir,  site)
        plot_path       = os.path.join(out_plot_dir, site)
        if save_videos:
            os.makedirs(output_vid_path, exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)

        print(f"Processing {site}: {site_idx + 1}/{len(SITES)}")
        pbar = tqdm(total=len(paired_files))

        # Per-patch accumulator for temporal depth plots (across all videos).
        bbox_results = {
            i: {"pred": [], "calb": [], "video": [], "frame": []}
            for i in range(len(site_ref_pts))
        }

        for vid_idx, (vid_file, depth_file) in enumerate(paired_files):
            vid_file_name = os.path.basename(vid_file)

            try:
                with open(os.path.join(animal_box_dir, site, f"{vid_file_name[:-4]}.json"), "r") as f:
                    animal_bboxes = json.load(f)
            except (FileNotFoundError, OSError):
                animal_bboxes = None

            raw_video         = cv2.VideoCapture(vid_file)
            frame_width       = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height      = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate        = raw_video.get(cv2.CAP_PROP_FPS)
            total_frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
            mask_dir_path     = os.path.join(mask_dir, site, vid_file_name[:-4])

            # Scale factors: annotation canvas → this video's actual frame size.
            scale_x = frame_width  / ANNOT_WIDTH
            scale_y = frame_height / ANNOT_HEIGHT
            scale_r = (scale_x + scale_y) / 2

            all_frame_preds = np.load(depth_file) if depth_file is not None else None

            # Carry the previous frame's calibration forward if the current
            # frame has too few visible circles to fit a reliable calibration.
            scale_prev, shift_prev = 1.0, 0.0

            # Accumulate depth maps for per-video variance plots.
            pred_stack = []
            calb_stack = []

            # Initialise video writer (only when --save-videos is set).
            # Frames are annotated in RGB but VideoWriter expects BGR, so we
            # convert before writing.  Output size matches the cropped frame.
            video_writer = None
            if save_videos:
                out_vid_file = os.path.join(output_vid_path,os.path.splitext(vid_file_name)[0] + '_annotated.mp4',)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(out_vid_file, fourcc, frame_rate,(frame_width, frame_height - CROP_BOTTOM),)

            for frame_idx in range(total_frame_count):
                frame = read_frame_safe(raw_video, site, vid_file_name, frame_idx)
                if frame is None:
                    break

                cropped_bgr   = (frame[:-CROP_BOTTOM, :, :]if frame.shape[0] > CROP_BOTTOM else frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB
                # BGR → RGB for annotation and plotting; convert back to BGR before writing video.
                cropped_frame = (frame[:-CROP_BOTTOM, :, :]if frame.shape[0] > CROP_BOTTOM else frame)

                if all_frame_preds is not None:
                    # Resize stored depth to the cropped frame resolution.
                    print("found pre-computed depth predictions; resizing to frame dimensions...")
                    raw_pred  = all_frame_preds[frame_idx]
                    pred      = cv2.resize(raw_pred, (frame_width, cropped_height), interpolation=cv2.INTER_LINEAR)
                else:
                    pred = get_depth(cropped_bgr, depth_model, depth_device, depth_args, depth_transform)
                pred_norm = normalize_depth_map(pred)

                # Load or generate the animal segmentation mask for this frame.
                frame_key = f"frame_{frame_idx:06d}"
                if animal_bboxes is not None:
                    animal_bboxes_frame = animal_bboxes.get(frame_key, [])
                else:
                    if _detection_model is None:
                        _detection_model = init_detection_model(depth_device)
                    animal_bboxes_frame = detect_animals(_detection_model, cropped_frame)

                animal_mask_frame   = load_animal_mask(frame_key, mask_dir_path, cropped_frame,
                    animal_bboxes_frame, mask_loader_fn=get_combined_sam_mask,)

                # Overlay SAM mask on frame for visual inspection (if available).
                if animal_mask_frame is not None and animal_mask_frame.sum():
                    frame = visualize_sam_mask_on_frame(cropped_frame, animal_mask_frame)
                # Convert back to BGR for annotation and video writing.  
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # ── Gather circle-patch training points ───────────────────────
                gt_depths_mean,   gt_depths_median = [], []
                gt_depths_all = []
                pred_disps_mean, pred_disps_median = [], []
                pred_disps_all = []

                for ref_point in site_ref_pts:
                    x, y, r, distance, unoccluded_pts = get_circle_info(
                        ref_point, animal_bboxes_frame,
                        scale_x, scale_y, scale_r, frame_shape,
                    )

                    if (x - r < 0 or y - r < 0
                            or x + r >= frame_width
                            or y + r >= frame_height):
                        continue
                    if len(unoccluded_pts) < 3:
                        continue

                    # Draw the circle and GT distance label on the frame.
                    cv2.circle(frame, (x, y), int(r), (0, 0, 255), 3)
                    label    = f"{distance:.1f}m"
                    text_pos = (x, max(y - int(r) - 10, 10))
                    cv2.putText(frame, label, text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 165, 255), 3, cv2.LINE_AA)

                    pred_vals = get_depth_on_circle(pred_norm, unoccluded_pts)
                    zs_vals   = projected_depth_from_distance(
                        distance, unoccluded_pts, u_map, v_map, FOCAL_LENGTH,
                    )

                    gt_depths_mean.append(np.nanmean(zs_vals))
                    gt_depths_median.append(np.nanmedian(zs_vals))
                    gt_depths_all.extend(zs_vals)
                    pred_disps_mean.append(np.nanmean(pred_vals))
                    pred_disps_median.append(np.nanmedian(pred_vals))
                    pred_disps_all.extend(pred_vals)

                # ── Fit per-frame disparity calibration ───────────────────────
                # Prefer median over mean; fall back to previous frame's params
                # if too few circles are visible.
                if len(pred_disps_median) > 2:
                    preds = np.array(pred_disps_median)
                    gts   = np.array(gt_depths_median)
                elif len(pred_disps_mean) > 2:
                    preds = np.array(pred_disps_mean)
                    gts   = np.array(gt_depths_mean)
                elif len(pred_disps_all) > 2:
                    preds = np.array(pred_disps_all)
                    gts   = np.array(gt_depths_all)
                else:
                    preds = gts = None

                if preds is not None and gts is not None:
                    scale, shift = align_disparity_scale_shift(preds, gts)
                    scale_prev, shift_prev = scale, shift
                else:
                    scale, shift = scale_prev, shift_prev  # carry forward

                # Apply calibration: disparity → depth.
                calb_disparity = pred_norm * scale + shift
                disp_safe      = np.where(np.abs(calb_disparity) < 1e-7,
                                          np.nan, calb_disparity)
                calb_depth     = 1.0 / disp_safe

                # ── Annotate animal distances and accumulate results ───────────
                frame = annotate_animal_distances(
                    frame, animal_bboxes_frame, calb_depth,
                    frame_shape, FOCAL_LENGTH,
                    combined_sam_mask=animal_mask_frame
                )

                if video_writer is not None:
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                update_bbox_results(
                    bbox_results, site_ref_pts,
                    scale_x, scale_y, scale_r,
                    pred_norm, calb_disparity,
                    vid_file_name, frame_idx,
                )

                pred_stack.append(pred_norm.copy())
                calb_stack.append(calb_disparity.copy())

            raw_video.release()
            if video_writer is not None:
                video_writer.release()

            # ── Per-video variance / VMR plots ────────────────────────────────
            if not pred_stack or not calb_stack:
                print(f"[WARN] No valid frames for {site}/{vid_file_name}. "
                      f"Skipping variance plots.")
                pbar.update(1)
                continue

            pred_stack_np = np.stack(pred_stack, axis=0).astype(np.float32)
            calb_stack_np = np.stack(calb_stack, axis=0).astype(np.float32)

            plot_prefix = os.path.join(plot_path, os.path.splitext(vid_file_name)[0])
            plot_depth_variance(pred_stack_np, calb_stack_np, plot_prefix)

            del pred_stack_np, calb_stack_np
            gc.collect()
            pbar.update(1)

        pbar.close()


if __name__ == "__main__":
    main()
