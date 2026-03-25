import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os, json
import torch
import time
from tqdm import tqdm

from midas.model_loader import default_models, load_model
from transformers import DPTImageProcessor, DPTForDepthEstimation


# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

first_execution = True



parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_path',
                    default=None,
                    help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                         'from camera)'
                    )

parser.add_argument('-o', '--output_path',default=None,help='Folder for output image')
parser.add_argument('-m', '--model_weights',default=None,help='Path to the trained weights of model')
parser.add_argument('-t', '--model_type',default='dpt_beit_large_512',
                    help='Model type: '
                         'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                         'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                         'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                         'openvino_midas_v21_small_256'
                    )

parser.add_argument('-s', '--side',action='store_true',help='Output images contain RGB and depth images side by side')

parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
parser.set_defaults(optimize=False)

parser.add_argument('--height',
                    type=int, default=None,
                    help='Preferred height of images feed into the encoder during inference. Note that the '
                         'preferred height may differ from the actual height, because an alignment to multiples of '
                         '32 takes place. Many models support only the height chosen during training, which is '
                         'used automatically if this parameter is not set.'
                    )
parser.add_argument('--square',
                    action='store_true',
                    help='Option to resize images to a square resolution by changing their widths when images are '
                         'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
                         'images is tried to be preserved if supported by the model.'
                    )
parser.add_argument('--grayscale',
                    action='store_true',
                    help='Use a grayscale colormap instead of the inferno one. Although the inferno colormap, '
                         'which is used by default, is better for visibility, it does not allow storing 16-bit '
                         'depth values in PNGs but only 8-bit ones due to the precision limitation of this '
                         'colormap.'
                    )
parser.add_argument('--outdir', type=str, default='../../outputs/depth_preds')
parser.add_argument('--data-dir', type=str, default='../../DATA/VIDEOS')

args = parser.parse_args()

# if args.model_weights is None:
#     args.model_weights = default_models[args.model_type]

# model, transform, net_w, net_h = load_model(device, args.model_weights, args.model_type, False, height=None,square=None)#,grayscale=False)
# model.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = None
args.model_type = "Huggingface"
image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512")
transform = image_processor
model.to(device).eval()

args.outdir = os.path.join(args.outdir, 'rltv_depths', 'MIDAS')
os.makedirs(args.outdir, exist_ok=True)

data_dir = args.data_dir
folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])


for k, site in enumerate(folders):

    pred_depth_dir = os.path.join(args.outdir,site)
    os.makedirs(pred_depth_dir, exist_ok=True)
    
    site_dir = os.path.join(data_dir,site)
    #filepaths = sorted([os.path.join(data_dir, site,f) for f in os.listdir(site_dir)])
    filepaths = sorted([os.path.join(data_dir, site, f) for f in os.listdir(site_dir) if f.upper().endswith((".AVI", ".MP4"))])

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
            raw_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

            #if args.model_type == "Huggingface":
            inputs = transform(images=raw_frame_rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
                # depth = torch.nn.functional.interpolate(
                #     predicted.unsqueeze(1),
                #     size=raw_frame_rgb.shape[:2],
                #     mode="bicubic",
                #     align_corners=False,
                # ).squeeze()
            predicted_depth = predicted_depth.squeeze().cpu().numpy()

            #image = transform({"image": raw_frame_rgb})["image"]
            #image = torch.from_numpy(image).to(device)

            # with torch.no_grad():
            #     raw_depth, depth = infer_image(device, model, args.model_type, image, (net_w, net_h), raw_frame_rgb.shape[1::-1],False, False)

            frame_depths.append(predicted_depth)

        raw_video.release()
        np_file_name = os.path.join(pred_depth_dir, base_file_name + '.npy')
        np.save(np_file_name, frame_depths)
        pbar.update(1)

    pbar.close()