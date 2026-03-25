"""
Estimate animal distances for the files listed in `DATA/Animal_Distances.xlsx`.

For each row, the script finds the matching image or video in
`DATA/animal_dist_imgs/`, loads the image or first video frame, calibrates the
depth prediction from the visible reference-point circles, and saves annotated
outputs plus an Excel summary report.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from PytorchWildlife.models import detection as pw_detection
from infer_utils import get_depth, initialize_model
from ultralytics.utils import LOGGER

from utils.analysis_utils import is_image_file, is_video_file

from utils.pipeline_utils import detect_animals, normalize_depth_map

from utils.calb_utils import (
    align_disparity_scale_shift,
    annotate_animal_distances,
    get_circle_info,
    get_combined_sam_mask,
    get_depth_on_circle,
    update_patch_stats,
)


ANNOT_HEIGHT = 4500
ANNOT_WIDTH = 8000
IMAGE_CROP_BOTTOM = 240
VIDEO_CROP_BOTTOM = 80
IMAGE_FL = 8777
VIDEO_FL = 2955
MIN_CALB_POINTS = 3
EPS = 1e-7

ANIMAL_DIST_DIR = "./DATA/animal_dist_imgs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate and save animal distances for files listed in an Excel sheet."
    )
    parser.add_argument(
        "--model-name",
        default="DA",
        help="Output subfolder name for this run.",
    )
    parser.add_argument(
        "--output-root",
        default="./outputs",
        help="Root directory for annotated outputs.",
    )
    parser.add_argument(
        "--excel-file",
        default=None,
        help="Excel sheet with columns: Site, File number, Distance.",
    )
    parser.add_argument(
        "--point-bbox-json",
        default=None,
        help="Patch/circle metadata JSON. Defaults to image/video dir candidate paths.",
    )
    parser.add_argument(
        "--ideal-calibration-json",
        default="./DATA/ideal_calibration_results.json",
        help="JSON with fallback site-wise scale/shift calibration.",
    )
    parser.add_argument(
        "--report-path",
        default="./REPORTS/animal_dist_rslt_DA.xlsx",
        help="Final Excel report output path.",
    )
    parser.add_argument(
        "--min-visible-patches",
        type=float,
        default=3.0,
        help="Use ideal calibration when visible patches are below this threshold.",
    )

    return parser.parse_args()


def configure_logging():
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    logging.getLogger("ultralytics").propagate = False
    LOGGER.setLevel(logging.ERROR)


def safe_normalize(depth):
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    max_val = float(np.max(finite))
    if max_val <= EPS:
        return np.zeros_like(depth, dtype=np.float32)
    return depth / max_val


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def find_source_file(file_number, site):
    """Find `{FileNumber}_{Site}.*` inside `DATA/animal_dist_imgs/`."""
    raw = clean_value(file_number)
    if not raw:
        return None
    stem = os.path.splitext(os.path.basename(raw))[0]
    if not stem:
        return None
    if not os.path.isdir(ANIMAL_DIST_DIR):
        return None
    target = f"{stem}_{site}".lower()
    for name in os.listdir(ANIMAL_DIST_DIR):
        name_stem, _ = os.path.splitext(name)
        if name_stem.lower() == target and (is_image_file(name) or is_video_file(name)):
            return os.path.join(ANIMAL_DIST_DIR, name)
    return None


def load_source_frame(file_path):
    file_name = os.path.basename(file_path)

    if is_image_file(file_name):
        frame = cv2.imread(file_path)
        if frame is None:
            return None, None
        return frame, "image"

    if is_video_file(file_name):
        # Distance sampling uses only the first occurrence, so first frame is enough.
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None, None
        return frame, "video"

    return None, None


def crop_frame(frame_bgr, media_type):
    crop_bottom = VIDEO_CROP_BOTTOM if media_type == "video" else IMAGE_CROP_BOTTOM
    fl = VIDEO_FL if media_type == "video" else IMAGE_FL
    if frame_bgr.shape[0] <= crop_bottom:
        return frame_bgr.copy(), fl
    return frame_bgr[:-crop_bottom, :, :], fl



def compute_geometry(frame_shape):
    height, width = frame_shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    c_x = width / 2.0
    c_y = height / 2.0
    return x - c_x, y - c_y


def compute_z_values(distance, circle_pts, u, v, fl):
    fl_sq = max(float(fl) ** 2, EPS)
    return [
        distance / np.sqrt(((u[py, px] ** 2 + v[py, px] ** 2) / fl_sq) + 1.0)
        for (px, py) in circle_pts
    ]


def collect_calibration_data(ref_pts,animal_bboxes_frame,pred_norm,scale_x,scale_y,scale_r,
    frame_shape,all_animal_mask_frame,u,v,fl,full_frame_shape=None,):

    gt_depth_mean = []
    gt_depth_median = []
    gt_depth_all = []
    gt_depth_outliers = []
    pred_depth_mean = []
    pred_depth_median = []
    pred_depth_all = []
    pred_depth_outliers = []
    visible_count = 0

    # Use full frame dims for outer bounds check (matches old code behaviour).
    bounds_height, bounds_width = full_frame_shape if full_frame_shape is not None else frame_shape

    for ref_point in ref_pts:
        x, y, r, distance, unoccluded_indices = get_circle_info(ref_point,animal_bboxes_frame,scale_x,
            scale_y,scale_r,frame_shape,all_animal_mask_frame,)

        if x - r < 0 or y - r < 0 or x + r >= bounds_width or y + r >= bounds_height:
            continue

        if len(unoccluded_indices) == 0:
            continue

        visible_count += 1
        pred_train = get_depth_on_circle(pred_norm, unoccluded_indices)
        zs_train = compute_z_values(distance, unoccluded_indices, u, v, fl)
        update_patch_stats(pred_train,zs_train,gt_depth_mean,gt_depth_median,gt_depth_all,
            gt_depth_outliers,pred_depth_mean,pred_depth_median,pred_depth_all,pred_depth_outliers,)

    return {
        "gt_mean": gt_depth_mean,
        "gt_median": gt_depth_median,
        "gt_all": gt_depth_all,
        "pred_mean": pred_depth_mean,
        "pred_median": pred_depth_median,
        "pred_all": pred_depth_all,
        "visible_count": visible_count,
    }


def pick_alignment_points(calb_data):
    candidates = (
        ("median", calb_data["pred_median"], calb_data["gt_median"]),
        ("all", calb_data["pred_all"], calb_data["gt_all"]),
        ("mean", calb_data["pred_mean"], calb_data["gt_mean"]),
    )

    for point_select, preds_raw, gts_raw in candidates:
        preds = np.array(preds_raw)
        gts = np.array(gts_raw)
        if len(preds) > 2 and len(gts) > 2:
            return point_select, preds, gts

    return None, None, None


def compute_calibration(site, calb_data, use_ideal_calb, ideal_calb):
    ideal = ideal_calb.get(site)
    if use_ideal_calb:
        if ideal is not None:
            return float(ideal["scale"]), float(ideal["shift"]), "ideal"
        print(f"[WARN] Ideal calibration missing for site {site}. Falling back to per-file calibration.")

    point_select, preds, gts = pick_alignment_points(calb_data)
    if preds is None:
        return float(ideal["scale"]), float(ideal["shift"]), "ideal"

    scale, shift = align_disparity_scale_shift(preds, gts)
    if not np.isfinite(scale) or not np.isfinite(shift):
        return float(ideal["scale"]), float(ideal["shift"]), "ideal"

    if scale < 0:
        scale = scale * -1

    return float(scale), float(shift), point_select


def append_summary_rows(results_df):
    if results_df.empty:
        return results_df

    mean_row = {
        "Site": "MEAN",
        "File name": "",
        "GT Dist": round(float(results_df["GT Dist"].mean()), 2),
        "Box Dist": round(float(results_df["Box Dist"].mean()), 2),
        "Mask Dist": round(float(results_df["Mask Dist"].mean()), 2),
        "diff_err": round(float(results_df["diff_err"].mean()), 2),
        "abs_err": round(float(results_df["abs_err"].mean()), 2),
        "abs_rel": round(float(results_df["abs_rel"].mean()), 2),
    }

    median_row = {
        "Site": "MEDIAN",
        "File name": "",
        "GT Dist": round(float(results_df["GT Dist"].median()), 2),
        "Box Dist": round(float(results_df["Box Dist"].median()), 2),
        "Mask Dist": round(float(results_df["Mask Dist"].median()), 2),
        "diff_err": round(float(results_df["diff_err"].median()), 2),
        "abs_err": round(float(results_df["abs_err"].median()), 2),
        "abs_rel": round(float(results_df["abs_rel"].median()), 2),
    }

    tod_rows = []
    tod_data = results_df[~results_df["Site"].isin(["MEAN", "MEDIAN"])].copy()
    if "ToD" in tod_data.columns:
        for tod, group in tod_data.groupby("ToD", dropna=True):
            vals = group["abs_rel"].to_numpy(dtype=np.float64)
            if vals.size == 0:
                continue
            tod_rows.append(
                {
                    "Site": f"ToD={tod}",
                    "File name": "",
                    "GT Dist": "",
                    "Box Dist": "",
                    "Mask Dist": "",
                    "diff_err": "",
                    "abs_err": "",
                    "abs_rel": round(float(np.median(vals)), 2),
                    "MAD_abs_rel": round(float(np.median(np.abs(vals - np.median(vals)))), 2),
                    "Mean_abs_rel": round(float(np.mean(vals)), 2),
                    "N": int(vals.size),
                }
            )

    return pd.concat([results_df, pd.DataFrame([mean_row, median_row] + tod_rows)],ignore_index=True,)


def save_annotated_frame(frame_rgb, output_dir, source_name, fl, tag="box"):
    output_subdir = os.path.join(output_dir)
    os.makedirs(output_subdir, exist_ok=True)
    output_name = f"{Path(source_name).stem}_dist_{int(fl)}_{tag}.JPG"
    output_path = os.path.join(output_subdir, output_name)
    cv2.imwrite(output_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    return output_path


def main():
    args = parse_args()
    configure_logging()

    _excel_file_ = os.path.join('./DATA', "Animal_Distances.xlsx")

    excel_file = args.excel_file or _excel_file_
    if excel_file is None:
        raise FileNotFoundError("Could not find Excel file.")

    ref_pts_path = os.path.join('./DATA', "reference_point_distance.json")

    if ref_pts_path is None:
        raise FileNotFoundError("Could not find circle patch JSON.")
    
    output_dir = os.path.join(args.output_root, 'animal_distance_outputs')


    depth_model_dict = initialize_model()
    depth_model = depth_model_dict["MODEL"].to(depth_model_dict["DEVICE"]).eval()
    device = depth_model_dict["DEVICE"]
    depth_args = depth_model_dict["ARGS"]
    depth_transform = depth_model_dict.get("TRANSFORM", None)

    detection_model = pw_detection.MegaDetectorV6(device=device,pretrained=True,version="MDV6-yolov10-e",)

    distance_df = pd.read_excel(excel_file)

    with open(ref_pts_path, "r") as f:
        ref_pts_data = json.load(f)
    with open(args.ideal_calibration_json, "r") as f:
        ideal_calb = json.load(f)

    animal_results = []
    row_iter = tqdm(distance_df.iterrows(), total=distance_df.shape[0], desc="Processing")
    
    for _, row in row_iter:
        site = clean_value(row.get("Site"))
        file_number = row.get("File number")

        if not site:
            continue
        if site not in ref_pts_data:
            print(f"[WARN] Patch metadata missing for site {site}. Skipping.")
            continue

        source_file = find_source_file(file_number, site)

        if source_file is None:
            print(f"[WARN] File not found for site={site}, file_number={file_number}. Skipping.")
            continue

        frame_bgr, media_type = load_source_frame(source_file)
        if frame_bgr is None:
            print(f"[WARN] Could not load frame from {source_file}. Skipping.")
            continue

        full_height, full_width = frame_bgr.shape[:2]

        cropped_bgr, fl = crop_frame(frame_bgr, media_type)
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        
        if cropped_bgr.size == 0:
            print(f"[WARN] Empty crop for {source_file}. Skipping.")
            continue

        frame_height, frame_width = cropped_bgr.shape[:2]
        frame_shape = (frame_height, frame_width)
        scale_x = full_width / ANNOT_WIDTH
        scale_y = full_height / ANNOT_HEIGHT
        scale_r = (scale_x + scale_y) / 2.0
        #u, v = compute_geometry((full_height, full_width))

        x, y = np.meshgrid(np.arange(full_width), np.arange(full_height))
        c_x = full_width/2
        c_y = full_height/2
        u = x - c_x
        v = y - c_y

        pred = get_depth(cropped_bgr, depth_model, device, depth_args, depth_transform)
        #np.save(os.path.join('DA_raw', f"{Path(source_file).stem}.npy"), pred)
        pred_norm = normalize_depth_map(pred)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        animal_bboxes_frame = detect_animals(detection_model, frame_rgb)
        all_animal_mask_frame = get_combined_sam_mask(cropped_rgb, animal_bboxes_frame)

        calb_data = collect_calibration_data(
            ref_pts=ref_pts_data[site],
            animal_bboxes_frame=animal_bboxes_frame,
            pred_norm=pred_norm,
            scale_x=scale_x,
            scale_y=scale_y,
            scale_r=scale_r,
            frame_shape=frame_shape,
            all_animal_mask_frame=all_animal_mask_frame,
            u=u,v=v,fl=fl,full_frame_shape=(full_height, full_width),)

        # patches_visible = row.get("Patches", np.nan)
        patches_visible_computed = calb_data["visible_count"]
        use_ideal_calb = patches_visible_computed < float(args.min_visible_patches)

        scale, shift, calb_source = compute_calibration(site=site,calb_data=calb_data,use_ideal_calb=use_ideal_calb,ideal_calb=ideal_calb,)

        Disparity = np.array(pred_norm) * scale + shift
        epsilon = 1e-7
        Disp_safe = np.where(np.abs(Disparity) < epsilon, np.nan, Disparity)
        # since we have converted our ground truth to Z which is Depth or Z
        depth_calb = 1.0 / Disp_safe # Z (Depth) = 1/Disparity

        gt_animal_distance = row.get("Distance", np.nan)
        gt_distance = float(gt_animal_distance) if pd.notna(gt_animal_distance) else None

        if gt_distance is None:
            annotated_rgb = annotate_animal_distances(
                cropped_bgr.copy(),
                animal_bboxes_frame,
                depth_calb,
                frame_shape,
                fl,
                gt_distance,
                full_frame_shape=(full_height, full_width),
            )
            animal_err = {}
        else:
            annotated_rgb_mask, annotated_rgb_box, animal_err = annotate_animal_distances(
                cropped_bgr.copy(),
                animal_bboxes_frame,
                depth_calb,
                frame_shape,
                fl,
                gt_distance,
                combined_sam_mask=all_animal_mask_frame,
                full_frame_shape=(full_height, full_width),
            )

        save_annotated_frame(annotated_rgb_mask, output_dir, os.path.basename(source_file), fl, tag="mask")
        save_annotated_frame(annotated_rgb_box, output_dir, os.path.basename(source_file), fl, tag="box")

        if not animal_err:
            continue

        min_idx = min(animal_err, key=lambda k: animal_err[k]["abs_rel"])
        min_err = animal_err[min_idx]
        animal_results.append(
            {
                "Site": site,
                "file_name": os.path.basename(source_file),
                "FL": fl,
                "Scale": scale,
                "Shift": shift,
                "Calibration": calb_source,
                "ToD": clean_value(row.get("ToD")) or np.nan,
                "GT Dist": gt_distance,
                "Box Dist": round(float(min_err["pred_box"]), 1),
                "Mask Dist": round(float(min_err["pred_mask"]), 1),
                "diff_err": round(float(min_err["diff_err"]), 2),
                "abs_err": round(float(min_err["abs_err"]), 2),
                "abs_rel": round(float(min_err["abs_rel"]), 2),
            }
        )

    results_df = pd.DataFrame(animal_results)
    results_df = append_summary_rows(results_df)

    report_dir = os.path.dirname(args.report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    results_df.to_excel(args.report_path, index=False)
    print(f"Saved report: {args.report_path}")
    print(f"Processed rows: {distance_df.shape[0]}, successful rows: {len(animal_results)}")


if __name__ == "__main__":
    main()
