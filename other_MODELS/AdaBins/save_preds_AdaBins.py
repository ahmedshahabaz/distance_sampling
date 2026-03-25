import argparse
import os,random
import json
import torch
import cv2
import numpy as np
import matplotlib
from tqdm import tqdm
from PIL import Image
from infer import InferenceHelper  # from AdaBins repo


# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def resize_keep_aspect_long_side(img_rgb: np.ndarray, long_side: int = 1280) -> np.ndarray:
    """
    Resize RGB image so that max(H, W) == long_side, keeping aspect ratio.
    """
    h, w = img_rgb.shape[:2]
    scale = long_side / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Best practice interpolation
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)

# ---------------------- Arguments ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='../../outputs/depth_preds')
parser.add_argument('--data-dir', type=str, default='../../DATA/VIDEOS')
args = parser.parse_args()

# ---------------------- AdaBins Setup ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
infer_helper = InferenceHelper(dataset='kitti')

# ---------------------- File Paths ----------------------    
args.outdir = os.path.join(args.outdir, 'metric_depths', 'ADABINS')
os.makedirs(args.outdir, exist_ok=True)


data_dir = args.data_dir
folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

# ---------------------- Frame-wise Inference ----------------------

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
            
            rgb_crop = resize_keep_aspect_long_side(raw_frame_rgb, long_side=966)
            
            # # Random‐crop to 704×352
            # h, w = raw_frame_rgb.shape[:2]
            # crop_h, crop_w = 352, 704
            # if h < crop_h or w < crop_w:
            #     raise ValueError(f"Frame too small ({h}×{w}) for crop {crop_h}×{crop_w}")
            # y0 = random.randint(0, h - crop_h)
            # x0 = random.randint(0, w - crop_w)
            # rgb_crop = raw_frame_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]


            # Turn into PIL and run inference
            pil_img = Image.fromarray(rgb_crop)

            with torch.no_grad():
                bin_centers, depth_pred = infer_helper.predict_pil(pil_img)
                depth_pred = depth_pred.squeeze()  # Shape: H x W

            frame_depths.append(depth_pred)

        raw_video.release()
        np_file_name = os.path.join(pred_depth_dir, base_file_name + '.npy')
        np.save(np_file_name, frame_depths)
        pbar.update(1)

    pbar.close()