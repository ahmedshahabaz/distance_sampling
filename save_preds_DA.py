"""
save_preds_DA.py – Run Depth Anything V2 inference and save raw depth predictions.

For each site, iterates over all .AVI videos, runs the model frame-by-frame,
and saves the full stack of per-frame depth arrays as a single .npy file:

    <outdir>/<site>/<video_stem>.npy   →  shape (num_frames, H, W), float32

The model is loaded from HuggingFace Hub:
    depth-anything/Depth-Anything-V2-Large-hf

Usage
-----
    python save_preds_DA.py [--outdir DIR] [--data-dir DIR]
"""

import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ── Constants ─────────────────────────────────────────────────────────────────

# Pixels to remove from the bottom of each frame (camera rig / timestamp strip).
CROP_BOTTOM = 80

_DEFAULT_DATA_DIR = './DATA/VIDEOS'
data_dir = './DATA/VIDEOS'
folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Save Depth Anything V2 per-frame depth predictions to .npy files.'
    )
    parser.add_argument(
        '--outdir', type=str,
        default='./outputs/depth_preds/rltv_depths/DA',
        help='Root output directory. Predictions are saved as <outdir>/<site>/<video>.npy',
    )
    parser.add_argument(
        '--data-dir', type=str,
        default=_DEFAULT_DATA_DIR,
        metavar='DIR',
        help='Directory containing site subdirectories with .AVI video files. (default: %(default)s)',
    )
    return parser.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device):
    """
    Load Depth Anything V2 Large from HuggingFace Hub.

    Returns
    -------
    model           : nn.Module (eval mode, on device)
    image_processor : AutoImageProcessor
    """
    hub_id = "depth-anything/Depth-Anything-V2-Large-hf"
    print(f"Loading model '{hub_id}' on {device}...")
    image_processor = AutoImageProcessor.from_pretrained(hub_id)
    model = AutoModelForDepthEstimation.from_pretrained(hub_id)
    model = model.to(device).eval()
    print("Model ready.\n")
    return model, image_processor


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = (
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )

    data_dir = args.data_dir
    model, image_processor = load_model(device)
    os.makedirs(args.outdir, exist_ok=True)

    # ── Site loop ─────────────────────────────────────────────────────────────
    for site_idx, site in enumerate(folders):
        site_output_dir = os.path.join(args.outdir, site)
        os.makedirs(site_output_dir, exist_ok=True)

        vid_files = sorted(
            os.path.join(data_dir, site, f)
            for f in os.listdir(os.path.join(data_dir, site))
            if f.upper().endswith('.AVI')
        )

        print(f'Processing {site}: {site_idx + 1}/{len(folders)}')
        pbar = tqdm(total=len(vid_files))

        # ── Video loop ────────────────────────────────────────────────────────
        for vid_file in vid_files:
            video_stem      = os.path.splitext(os.path.basename(vid_file))[0]
            output_npy_path = os.path.join(site_output_dir, video_stem + '.npy')

            raw_video         = cv2.VideoCapture(vid_file)
            total_frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_depths = []

            # ── Frame loop ────────────────────────────────────────────────────
            for frame_idx in range(total_frame_count):
                ret, raw_frame = raw_video.read()
                if not ret or raw_frame is None:
                    print(f"[WARN] Bad frame {frame_idx} in {video_stem}. Stopping early.")
                    break

                # Crop the bottom strip before inference.
                cropped = (raw_frame[:-CROP_BOTTOM, :, :]if raw_frame.shape[0] > CROP_BOTTOM else raw_frame)
                frame_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

                with torch.no_grad():
                    inputs = image_processor(images=frame_rgb,return_tensors="pt").to(device)
                    predicted_depth = model(**inputs).predicted_depth
                    predicted_depth = predicted_depth.squeeze().cpu().numpy()

                frame_depths.append(predicted_depth)

            raw_video.release()

            # Save all frames for this video as a single array (num_frames, H, W).
            np.save(output_npy_path, frame_depths)
            pbar.update(1)

        pbar.close()


if __name__ == "__main__":
    main()
