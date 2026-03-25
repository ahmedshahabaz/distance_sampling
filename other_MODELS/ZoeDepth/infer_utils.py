import argparse
import os
import json
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
from tqdm import tqdm
from PIL import Image
import types

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import pil_to_batched_tensor

from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation, ZoeDepthConfig
from transformers import DPTImageProcessor, DPTForDepthEstimation


# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def initialize_model(dataset=None):
    midas_model_type = "DPT_Large"
    torch.hub.help("intel-isl/MiDaS", "DPT_Large", force_reload=True)  # Triggers fresh download of MiDaS repo

    # ---------------------- Arguments ----------------------
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = None
 
    repo = "isl-org/ZoeDepth"
    # conf = get_config("zoedepth_nk", mode="eval", pretrained=True, , midas_model_type=midas_model_type)  # Change this to other variants if needed
    # model = build_model(conf).to(device)

    # # Zoe_NK
    # model = torch.hub.load(repo, "ZoeD_NK", pretrained=True, config_mode="eval")#,midas_model_type=midas_model_type)
    # model = torch.hub.load('other_MODELS/ZoeDepth/', "ZoeD_NK", source='local',pretrained=True, config_mode="eval",midas_model_type=midas_model_type)

    
    if dataset is None:
        image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        #configuration = ZoeDepthConfig(use_pretrained_backbone=False)
        #print(configuration)
        model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")
        #model = ZoeDepthForDepthEstimation.from_pretrained(configuration)

    elif dataset=='nyu':
        image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu")
        model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu")

    elif dataset=='kitti':
        image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-kitti")
        model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-kitti")


    return {'MODEL':model, 'DEVICE': device, 'ARGS': args, 'TRANSFORM':image_processor,'net_w': None, 'net_h': None}


def get_depth(img,model,device,args=None,transform=None,net_w=None,net_h=None):


    if transform == None:
        pil_img = Image.fromarray(img)

        with torch.no_grad():
            X = pil_to_batched_tensor(pil_img).to(device)
            depth_tensor = model.infer(X)
            depth = depth_tensor.squeeze().cpu().numpy()

    if transform is not None:

        with torch.no_grad():   
            inputs = transform(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

            # interpolate to original size
            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img.shape[:2],                
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

    return depth