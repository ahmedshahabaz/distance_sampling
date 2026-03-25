import argparse
import os
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
infer_helper = None

def initialize_model(dataset='kitti'):
    global infer_helper
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_helper = InferenceHelper(dataset=dataset)

    image_processor = None

    return {'MODEL':None, 'DEVICE': device, 'ARGS': args, 'TRANSFORM':image_processor,'net_w': None, 'net_h': None}


def get_depth(img,model,device,args,transform=None,net_w=None,net_h=None):

    with torch.no_grad():
        #img = torch.from_numpy(img).float().to(device)
        #img = img.permute(2, 0, 1).unsqueeze(dim=0)
        pil_image = Image.fromarray(img)
        bin_centers, depth_pred = infer_helper.predict_pil(img)
        depth_pred = depth_pred.squeeze()  # Shape: H x W

    return depth_pred