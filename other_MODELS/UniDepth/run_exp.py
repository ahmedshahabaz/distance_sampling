import argparse
import os
import cv2
import torch
import json, matplotlib
import numpy as np
from tqdm import tqdm
from PIL import Image
from unidepth.models import UniDepthV2

# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# ---------------------- Arguments ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='/mnt/wwn-0x5000c500e59fc407/Datasets/Animal/output/model_preds')
args = parser.parse_args()

# ---------------------- UniDepth Setup ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "unidepth-v2-vitl14"
model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}", strict=False).to(device)

# ---------------------- File Paths ----------------------  
args.outdir = os.path.join(args.outdir, 'UNIDEPTH')
os.makedirs(args.outdir, exist_ok=True)

#data_dir = '/home/mithrandir/Documents/PhD/Projects/animal/data/wolf/test_ds'
data_dir = '/home/mithrandir/Documents/PhD/Projects/animal/Depth-Anything-V2/DATA/exp/vids'
npy_dir =  '/home/mithrandir/Documents/PhD/Projects/animal/Depth-Anything-V2/DATA/exp/depth/UNIDEPTH'
os.makedirs(npy_dir, exist_ok=True)
#data_dir = './input_vid'

files = os.listdir(data_dir)
filenames = sorted([os.path.join(data_dir, f) for f in files if f.endswith(".mp4")])
#filenames = sorted([os.path.join(data_dir, f) for f in files if f.endswith(".mp4")])

margin_width = 50
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

json_file_path = "../DATA/exp/vids/bbox_point_distance.json"

# Load bounding box JSON file
with open(json_file_path, "r") as f:
    bbox_data = json.load(f)

for k, filename in enumerate(filenames):
    print(f'Processing {k+1}/{len(filenames)}: {filename}')

    raw_video = cv2.VideoCapture(filename)
    frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))

    base_file_name = os.path.splitext(os.path.basename(filename))[0]
    output_path = os.path.join(args.outdir, base_file_name + '_depth.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width * 2, frame_height))

    frame_depths = []
    bboxes = bbox_data.get(str(k+1), [])

    pbar = tqdm(total=total_frame_count)

    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break

        # Convert OpenCV BGR frame to RGB and prepare for UniDepth
        raw_frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        input_image = torch.from_numpy(np.array(Image.fromarray(raw_frame_rgb))).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: (1, C, H, W)

        # Run inference with UniDepth
        with torch.no_grad():
            predictions = model.infer(input_image)
            depth_pred = predictions["depth"].squeeze().cpu().numpy()  # Metric depth map

        # Normalize depth for visualization
        norm_depth = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min()) * 255.0
        norm_depth = depth.astype(np.uint8)
        #depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)  # Apply color mapping
        depth_colormap = (cmap(norm_depth)[..., :3] * 255).astype(np.uint8)

        # Store frame depth for saving
        frame_depths.append(depth_pred)

        # Draw bounding boxes
        for bbox_id, bbox_info in enumerate(bboxes):
            x1, y1, x2, y2 = bbox_info["bounding_box"]
            cv2.rectangle(raw_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(raw_frame, str(bbox_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Concatenate original and depth frames side by side
        combined_frame = cv2.hconcat([raw_frame, depth_colormap])
        out.write(combined_frame)

        pbar.update(1)

    # Save frame depths as NumPy array
    np_file_name = os.path.join(npy_dir, base_file_name + '_depth.npy')
    np.save(np_file_name, frame_depths)

    raw_video.release()
    out.release()
    pbar.close()
