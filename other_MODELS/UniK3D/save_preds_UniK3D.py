import argparse
import os
import cv2
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
from PIL import Image
from unik3d.models import UniK3D

# ---------------------- Arguments ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='../../outputs/depth_preds')
parser.add_argument('--data-dir', type=str, default='../../DATA/VIDEOS')
args = parser.parse_args()

# ---------------------- Model Setup ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(device)
model.eval()

# ---------------------- Directory Setup ----------------------
args.outdir = os.path.join(args.outdir, 'metric_depths', 'UNIK3D')
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
            
            rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            rgb_tensor = torch.from_numpy(np.array(Image.fromarray(rgb_frame))).permute(2, 0, 1).to(device) # (C, H, W)

            with torch.no_grad():
                prediction = model.infer(rgb_tensor)
                depth = prediction["depth"].squeeze().cpu().numpy()
            
            frame_depths.append(depth)
            #np_file_name = os.path.join(vid_depth_dir, f'frame_{frame_idx:06d}' + '.npy')
            #np.save(np_file_name, depth)
            #frame_idx += 1

        raw_video.release()
        np_file_path = os.path.join(pred_depth_dir, base_file_name + '.npy')
        np.save(np_file_path, frame_depths)
        pbar.update(1)
    
    pbar.close()
