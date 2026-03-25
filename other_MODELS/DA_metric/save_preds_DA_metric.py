import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from PIL import Image
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from depth_anything_v2.dpt import DepthAnythingV2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation




parser = argparse.ArgumentParser(description='Depth Anything V2')

#parser.add_argument('--video-path', type=str, default='/home/mithrandir/Documents/PhD/Projects/Depth Animal/data/wolf')
parser.add_argument('--input-size', type=int, default=518)
#parser.add_argument('--outdir', type=str, default='/home/mithrandir/Documents/PhD/Projects/animal/output/depths')
#parser.add_argument('--outdir', type=str, default='/mnt/wwn-0x5000c500e59fc407/Datasets/Animal/Distance_Sampling/DS_project/depth_preds')
parser.add_argument('--outdir', type=str, default='../../outputs/depth_preds')
parser.add_argument('--data-dir', type=str, default='../../DATA/VIDEOS')

parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
#parser.add_argument('--load-from', type=str, default='./checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
parser.add_argument('--load-from', type=str, default='./checkpoints/depth_anything_v2_metric_vkitti_vitl.pth')

parser.add_argument('--max-depth', type=float, default=20)

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

#depth_anything = DepthAnythingV2(**model_configs[args.encoder])

transform = None
depth_anything = DepthAnythingV2(**{**model_configs[args.encoder]})
ckpt_path = os.path.join(args.load_from)
depth_anything.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

#configuration = DepthAnythingConfig(depth_estimation_type='metric',max_depth=20)
# image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
# depth_anything = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
# transform = image_processor

depth_anything = depth_anything.to(DEVICE).eval()

args.outdir = os.path.join(args.outdir, 'metric_depths', 'DA')
os.makedirs(args.outdir, exist_ok=True)

data_dir = args.data_dir
folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

# ---------------------- Video-wise Inference ----------------------

for k, site in enumerate(folders):

    pred_depth_dir = os.path.join(args.outdir,site)
    os.makedirs(pred_depth_dir, exist_ok=True)

    site_dir = os.path.join(data_dir,site)
    #filepaths = sorted([os.path.join(data_dir, site_number,f) for f in os.listdir(site_dir)])
    filepaths = sorted([os.path.join(data_dir, site, f) for f in os.listdir(site_dir) if f.upper().endswith((".AVI", ".MP4"))])
    #depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir)])

    print(f'Processing {site}: {k+1}/{len(folders)}')
    pbar = tqdm(total=len(filepaths))

    for vid_file_path in filepaths:
        base_file_name = os.path.splitext(os.path.basename(vid_file_path))[0]
        vid_depth_dir = os.path.join(pred_depth_dir,base_file_name)
        os.makedirs(vid_depth_dir, exist_ok=True)
        
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

            if transform is not None:
                with torch.no_grad():
                    inputs = transform(images=raw_frame_rgb,return_tensors="pt").to(DEVICE)
                    outputs = depth_anything(**inputs)
                    predicted_depth = outputs.predicted_depth
                    predicted_depth = predicted_depth.squeeze().cpu().numpy()

                    # interpolate to original size
                    # depth = torch.nn.functional.interpolate(
                    #     predicted_depth.unsqueeze(1),
                    #     size=raw_frame_rgb.shape[:2],                
                    #     mode="bicubic",
                    #     align_corners=False,
                    # ).squeeze().cpu().numpy()


            else:
                predicted_depth, _ = depth_anything.infer_image(cropped_frame, args.input_size)

            #np_file_name = os.path.join(vid_depth_dir, f'frame_{frame_idx:06d}' + '.npy')
            #np.save(np_file_name, predicted_depth)

            frame_depths.append(predicted_depth)

        raw_video.release()
        np_file_name = os.path.join(pred_depth_dir, base_file_name + '.npy')
        np.save(np_file_name, frame_depths)
        pbar.update(1)

    pbar.close()
