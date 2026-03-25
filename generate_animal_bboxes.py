# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Run animal detection on all videos in `DATA/VIDEOS`.

For each video, this script saves:
  - per-frame detection JSON to `./outputs/animal_bboxes/<site>/`
  - an annotated preview video to `./outputs/detected_vids/<site>/`
"""
#%% 
# Importing necessary basic libraries and modules
from PIL import Image
import numpy as np
import supervision as sv
import json

#%% 
# PyTorch imports for tensor operations
import argparse
import torch
import os
#%% 
# Importing the models, transformations, and utility functions from PytorchWildlife 
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife import utils as pw_utils

import logging

# Silence ultralytics logger output (per-frame "0: 736x1280..." + "Speed: ...")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("ultralytics").propagate = False

# Also silence the main Ultralytics LOGGER object (used internally)
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.ERROR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initializing the model for image detection

#detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True, version="a")

detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="MDV6-yolov10-e")

# Initializing the model for image classification
#classification_model = pw_classification.AI4GOpossum(device=DEVICE, pretrained=True)
#classification_model = pw_classification.AI4GSnapshotSerengeti(device=DEVICE, pretrained=True)
classification_model = pw_classification.AI4GAmazonRainforest(device=DEVICE, version='v2')

# Defining transformations for detection and classification
# trans_det = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,stride=detection_model.STRIDE)
# trans_clf = pw_trans.Classification_Inference_Transform(target_size=224)

# Initializing a box annotator for visualizing detections
box_annotator = sv.BoxAnnotator(thickness=4)
lab_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=4, text_scale=2)


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results_det = detection_model.single_image_detection(frame, img_path=index)
    labels = []
    for xyxy in results_det["detections"].xyxy:
        cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
        results_clf = classification_model.single_image_classification(cropped_image)
        labels.append("{} {:.2f}".format(results_clf["prediction"], results_clf["confidence"]))
    annotated_frame = lab_annotator.annotate(
        scene=box_annotator.annotate(
            scene=frame,
            detections=results_det["detections"],
        ),
        detections=results_det["detections"],
        labels=labels,
    )

    for _det_, _label_ in zip(results_det['detections'],results_det['labels']):
        detection_data = {"x1": int(_det_[0][0]),"y1": int(_det_[0][1]),"x2": int(_det_[0][2]),"y2": int(_det_[0][3]),
            "confidence": float(_det_[2]), "mask":_det_[1], 'label':_label_, 'frame_num':index}

        frame_key  = f"frame_{index:06d}"
        all_detections.setdefault(frame_key, []).append(detection_data)

    return annotated_frame


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./DATA/VIDEOS')
parser.add_argument('--output-dir', type=str, default='./outputs/animal_bboxes')
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir

vid_fps = 20

all_detections = {}

folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

for k, site in enumerate(folders):

    camera_dir = os.path.join(data_dir,site)
    video_paths = sorted([os.path.join(camera_dir,f) for f in os.listdir(camera_dir)])

    os.makedirs(os.path.join('./outputs/', 'detected_vids', site), exist_ok=True)
    os.makedirs(os.path.join(output_dir, site), exist_ok=True)

    print(f'Processing {site}: {k+1}/{len(folders)}')

    for vid_path in video_paths:
        
        in_video_name = os.path.basename(vid_path)
        out_video_name = in_video_name[:-4]+'_bbox.mp4'
        json_flile_name = in_video_name[:-4]+'.json'
        json_file_path = os.path.join(output_dir, site, json_flile_name)

        TARGET_VIDEO_PATH = os.path.join('./outputs/', 'detected_vids', site, out_video_name)
        #TARGET_VIDEO_PATH = None
        # Processing the video and saving the result with annotated detections and classifications
        pw_utils.process_video(source_path=vid_path, target_path=TARGET_VIDEO_PATH, callback=callback, target_fps=vid_fps)

        #np.save("detection.npy", _detection_ar_)
        with open(json_file_path, 'w') as f:
            json.dump(all_detections, f)

        all_detections.clear()
