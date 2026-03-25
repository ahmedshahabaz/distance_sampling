import argparse
import os
import torch
import numpy as np
from PIL import Image
from unik3d.models import UniK3D

# Set torch backend optimizations
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def initialize_model(dataset=None):
    """
    Initializes the UniK3D model and relevant settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='/mnt/wwn-0x5000c500e59fc407/Datasets/Animal/Distance_Sampling/DS_project/depth_preds')
    args, unknown = parser.parse_known_args()

    net_w = None
    net_h = None
    transform = None  # UniK3D handles normalization internally
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load UniK3D model
    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(device)
    model.eval()

    return {
        'MODEL': model,
        'DEVICE': device,
        'ARGS': args,
        'TRANSFORM': transform,
        'net_w': net_w,
        'net_h': net_h
    }


def get_depth(img, model, device, args, transform=None, net_w=None, net_h=None):
    """
    Runs UniK3D inference on a single RGB image (numpy array).
    """
    # Convert image to tensor format (C, H, W)
    input_tensor = torch.from_numpy(np.array(Image.fromarray(img))).permute(2, 0, 1).to(device)

    # Run inference
    with torch.no_grad():
        predictions = model.infer(input_tensor)
        depth_pred = predictions["depth"].squeeze().cpu().numpy()

    return depth_pred
