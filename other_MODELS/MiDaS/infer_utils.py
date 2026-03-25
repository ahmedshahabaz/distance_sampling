import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os, json
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from transformers import DPTImageProcessor, DPTForDepthEstimation

from .midas.model_loader import default_models, load_model


# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

first_execution = True

@torch.no_grad()
def infer_image(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if isinstance(image, torch.Tensor):
        sample = image.to(device)
    elif isinstance(image, np.ndarray):
        sample = torch.from_numpy(image).to(device)

    if sample.ndim == 3:
        sample = sample.unsqueeze(0) 
    #sample = torch.from_numpy(image).to(device).unsqueeze(0)

    # if optimize and device == torch.device("cuda"):
    #     # if first_execution:
    #     #     print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
    #     #           "  float precision to work properly and may yield non-finite depth values to some extent for\n"
    #     #           "  half-floats.")
    #     #sample = sample.to(memory_format=torch.channels_last)
    #     #sample = sample.half()


    raw_prediction = model(sample)
    prediction = (
        F.interpolate(raw_prediction.unsqueeze(1),size=target_size[::-1],mode="bicubic",
        align_corners=False,)
        .squeeze()
        .cpu()
        .numpy()
    )

    #plt.imshow(prediction)

    return raw_prediction.squeeze().cpu().numpy(), prediction

def colorize_depth(depth):
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    colored_depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return colored_depth


def initialize_model():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)')

    parser.add_argument('-o', '--output_path',default=None,help='Folder for output image')
    parser.add_argument('-m', '--model_weights',default=None,help='Path to the trained weights of model')
    parser.add_argument('-t', '--model_type',default='dpt_beit_large_512', help='Model type: '
                             'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                             'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                             'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                             'openvino_midas_v21_small_256')

    args, unknown = parser.parse_known_args()
    
    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_w = None
    net_h = None


    #args.model_type = 'dpt_beit_large_512'
    #model, transform, net_w, net_h = load_model(device, args.model_weights, args.model_type, False, height=None,square=False)#,grayscale=False)
    #return {'MODEL':model, 'DEVICE': device, 'ARGS': args, 'TRANSFORM':transform,'net_w': net_w, 'net_h': net_h}
    
    #args.model_type = "DPT_Large"
    #args.model_type = "MiDaS_small"
    #args.model_type = "MiDaS"
    #args.model_type = "DPT_BEiT_L_512"

    #model = torch.hub.load("intel-isl/MiDaS", args.model_type)
    #midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    args.model_type = "Huggingface"

    # processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
    # model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512")

    #model.to(device)
    #model.eval()

    if args.model_type == "Huggingface":
        processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512")

        transform = processor  # Use processor as transform
        return {'MODEL': model, 'DEVICE': device, 'ARGS': args, 'TRANSFORM': transform, 'net_w': net_w, 'net_h': net_h}

    
    if args.model_type=="DPT_BEiT_L_512":
        transform = midas_transforms.beit512_transform

    if args.model_type == "DPT_Large" or args.model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform

    else:
        transform = midas_transforms.small_transform


    return {'MODEL':model, 'DEVICE': device, 'ARGS': args, 'TRANSFORM':transform,'net_w': net_w, 'net_h': net_h}


def get_depth(img,model,device,args,transform=None,net_w=None,net_h=None):

    # img is RGB image. MIDAS expects RGB input

    #raw_frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if args.model_type == "Huggingface":
        inputs = transform(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            return prediction.cpu().numpy()

    if transform is not None:

        if args.model_type=="DPT_BEiT_L_512" or args.model_type == "DPT_Large" or args.model_type == "MiDaS_small" or args.model_type == "MiDaS":
            # using torch hub model
            image = transform(img)

        else:
            image = transform({"image": img})["image"]
    #image = torch.from_numpy(image).to(device)
    with torch.no_grad():
        raw_depth, depth = infer_image(device, model, args.model_type, image, (net_w, net_h), img.shape[1::-1],False, False)

    return depth

