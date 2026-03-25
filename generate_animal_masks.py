"""
Generate and cache per-frame animal segmentation masks from saved bbox JSON.

Expected input:
  `./outputs/animal_bboxes/<site>/<video>.json`

Output:
  `./outputs/animal_seg_mask/<site>/<video>/<frame>.png`
"""

import argparse
import cv2
from cv2 import ellipse2Poly
import glob
import matplotlib
import numpy as np
import os, random, sys
import json
from tqdm import tqdm
from collections import defaultdict

from utils.calb_utils import get_combined_sam_mask

if __name__ == '__main__':

    data_dir = './DATA/VIDEOS'    
    output_dir = "./outputs/animal_seg_mask"
    animal_box_dir = './outputs/animal_bboxes'

    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    annot_height, annot_width = 4500, 8000

    # width = 2688, height = 1520
    frame_width, frame_height = 2688, 1520
    cropped_height = frame_height - 80
    frame_shape = (cropped_height,frame_width)

    for k, site in enumerate(folders):

        # depth_dir = os.path.join(output_dir,site)

        camera_dir = os.path.join(data_dir,site)
        filepaths = sorted([os.path.join(camera_dir,f) for f in os.listdir(camera_dir)])

        print(f'Processing {site}: {k+1}/{len(folders)}')
        pbar = tqdm(total=len(filepaths))
        
        for _vid_idx_, vid_file in enumerate(filepaths):

            _vid_file_name_ = os.path.basename(vid_file)

            mask_out_dir = os.path.join(output_dir, site,_vid_file_name_[:-4])
            os.makedirs(mask_out_dir, exist_ok=True)

            animal_box_path = os.path.join(animal_box_dir, site, f'{_vid_file_name_[:-4]}.json')

            with open(animal_box_path, 'r') as f:
                animal_bboxes = json.load(f)

            raw_video = cv2.VideoCapture(vid_file)
            # # width = 2688, height = 1520
            # frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # cropped_height = frame_height - 80
            frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
            total_frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))

            scale_x = frame_width / annot_width
            scale_y = frame_height / annot_height
            scale_r = (scale_x + scale_y) / 2
            
            for frame_idx in range(total_frame_count):

                ret, frame = raw_video.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cropped_frame = frame[:-80, :, :] if frame.shape[0] > 80 else frame

                # Reuse cached detections so SAM does not need to rediscover animals.
                frame_key = f"frame_{frame_idx:06d}"
                animal_bboxes_frame = animal_bboxes.get(frame_key, [])  # Get list of animal bounding boxes, empty if none
                #print(animal_bboxes_frame)
                all_animal_mask_frame = get_combined_sam_mask(cropped_frame,animal_bboxes_frame)

                mask_filename = f"{frame_idx:06d}.png"
                mask_path = os.path.join(mask_out_dir, mask_filename)

                cv2.imwrite(mask_path, (all_animal_mask_frame * 255).astype(np.uint8))

            raw_video.release()
            pbar.update(1)
        pbar.close()



