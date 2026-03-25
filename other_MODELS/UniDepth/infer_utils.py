import argparse
import os
import cv2
import torch
import json, matplotlib
import numpy as np
from tqdm import tqdm
from PIL import Image
from .unidepth.models import UniDepthV2


# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def initialize_model(dataset=None):

    # ---------------------- Arguments ----------------------
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()

    net_w = None
    net_h = None
    transform = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- UniDepth Setup ----------------------
    # model_name = "unidepth-v2-vitl14"
    # model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}").to(device)
    
    version = "v2"
    backbone = "vitl14"
    model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=version, backbone=backbone, pretrained=True, trust_repo=True, force_reload=True)

    return {'MODEL':model, 'DEVICE': device, 'ARGS': args, 'TRANSFORM':transform,'net_w': net_w, 'net_h': net_h}


def get_depth(img,model,device,args,transform=None,net_w=None,net_h=None):

    input_image = torch.from_numpy(np.array(Image.fromarray(img))).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: (1, C, H, W)

    # Run inference with UniDepth
    with torch.no_grad():
        predictions = model.infer(input_image)
        depth_pred = predictions["depth"].squeeze().cpu().numpy()  # Metric depth map

    return depth_pred