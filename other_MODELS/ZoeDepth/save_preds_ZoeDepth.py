import argparse
import os
import json
import torch
import cv2
import numpy as np
import matplotlib
from tqdm import tqdm
from PIL import Image

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import pil_to_batched_tensor

from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation, ZoeDepthConfig

# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.hub.help("intel-isl/MiDaS", "DPT_Large", force_reload=True)  # Triggers fresh download of MiDaS repo

# ---------------------- Arguments ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='../../outputs/depth_preds')
parser.add_argument('--data-dir', type=str, default='../../DATA/VIDEOS')
args = parser.parse_args()

# ---------------------- ZoeDepth Setup ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = None
# conf = get_config("zoedepth", "infer")  # Change this to other variants if needed
# model = build_model(conf).to(device)

image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")
transform = image_processor
model = model.to(device)
model.eval()

# ---------------------- File Paths ----------------------    
args.outdir = os.path.join(args.outdir, 'metric_depths', 'ZOEDEPTH')
os.makedirs(args.outdir, exist_ok=True)

data_dir = args.data_dir
folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

# ---------------------- Video-wise Inference ----------------------

for k, site in enumerate(folders):

    pred_depth_dir = os.path.join(args.outdir,site)
    os.makedirs(pred_depth_dir, exist_ok=True)
    
    site_dir = os.path.join(data_dir,site)
    #filepaths = sorted([os.path.join(data_dir, site,f) for f in os.listdir(site_dir)])
    filepaths = sorted([os.path.join(data_dir, site, f) for f in os.listdir(site_dir) if f.upper().endswith((".AVI", ".MP4"))])
    #depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir)])

    print(f'Processing {site}: {k+1}/{len(folders)}')
    pbar = tqdm(total=len(filepaths))

    for vid_file_path in filepaths:
        base_file_name = os.path.splitext(os.path.basename(vid_file_path))[0]
        vid_depth_dir = os.path.join(pred_depth_dir, base_file_name)
        #os.makedirs(vid_depth_dir, exist_ok=True)

        raw_video = cv2.VideoCapture(vid_file_path)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cropped_height = frame_height - 80
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        total_frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_depths = []
        frame_idx = 0

        for frame_idx in range(total_frame_count):
            ret, raw_frame = raw_video.read()
            if not ret or raw_frame is None:
                if frame_idx < total_frame_count:
                    print("Bad frame at index", frame_idx, "in", base_file_name)
                break

            cropped_frame = raw_frame[:-80, :, :] if raw_frame.shape[0] > 80 else raw_frame

            raw_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(raw_frame_rgb)

            if transform is None:
                with torch.no_grad():
                    X = pil_to_batched_tensor(pil_img).to(device)
                    depth_tensor = model.infer(X)
                    predicted_depth = depth_tensor.squeeze().cpu().numpy()

            elif transform is not None:
                with torch.no_grad():   
                    inputs = transform(images=raw_frame_rgb, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                    predicted_depth = predicted_depth.squeeze().cpu().numpy()

                    # interpolate to original size
                    # depth = torch.nn.functional.interpolate(
                    #     predicted_depth.unsqueeze(1),
                    #     size=raw_frame_rgb.shape[:2],                
                    #     mode="bicubic",
                    #     align_corners=False,
                    # ).squeeze().cpu().numpy()

            frame_depths.append(predicted_depth)

        raw_video.release()
        np_file_name = os.path.join(pred_depth_dir, base_file_name + '.npy')
        np.save(np_file_name, frame_depths)
        pbar.update(1)

    pbar.close()