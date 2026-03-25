import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os, json
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def colorize_depth(depth):
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    colored_depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return colored_depth


def initialize_model(dataset=None):

    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--input-size', type=int, default=518)

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='./checkpoints/depth_anything_v2_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    args, unknown = parser.parse_known_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    image_processor = None

    # depth_anything = DepthAnythingV2(**{**model_configs[args.encoder]})
    # checkpoint = torch.load(args.load_from, map_location='cpu')
    # state_dict = checkpoint
    # depth_anything.load_state_dict(state_dict,strict=True)

    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    depth_anything = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")


    return {'MODEL':depth_anything, 'DEVICE': device, 'ARGS': args, 'TRANSFORM':image_processor,'net_w': None, 'net_h': None}


def get_depth(img,model,device,args,transform=None,net_w=None,net_h=None):

    if transform is not None:
        with torch.no_grad():
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inputs = transform(images=img_rgb,return_tensors="pt").to(device)
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

            # interpolate to original size
            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img.shape[:2],                
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

    else:
        raw_depth, depth = model.infer_image(img, args.input_size)

    return depth


