import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import sys

model_name_dict = {'MD': "MIDAS", 'DA': "DepthAnything", 'ZD': "ZoeDepth", 'UD': "UniDepth", 'UN': "UniK3D", 'AB': "AdaBins"}

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("❗ Usage: python eval_kitti.py <model_name> <use_relative>")
    print("    <model_name> should be one of: MD, DA, ZD, UD, AB, UN")
    print("    <use_relative> should be: True or False (for relative depth evaluation)")
    print("    [data_split]   : optional, one of val, test, eigen (default: eigen)")
    sys.exit(1)

model_name = sys.argv[1]
use_relative_input = sys.argv[2]
data_split = sys.argv[3].lower() if len(sys.argv) == 4 else "eigen"

if data_split.lower() not in ("val", "test", "eigen", "validation", "valid"):
    print("❗ Error: data_split must be one of: val, test, eigen")
    sys.exit(1)

if data_split.lower() in ['validation', 'valid']:
    data_split = 'val'

# Parse boolean string to actual boolean
if use_relative_input.lower() in ['true', '1', 'yes']:
    use_relative = True
elif use_relative_input.lower() in ['false', '0', 'no']:
    use_relative = False
else:
    print("❗ Error: <use_relative> must be either 'True' or 'False'")
    sys.exit(1)

dataset = None

if model_name=='MD':
    midas_dir = os.path.join(os.path.dirname(__file__), 'other_MODELS', 'MiDaS')
    sys.path.insert(0, midas_dir)
    from other_MODELS.MiDaS.infer_utils import *

if model_name=='DA' and use_relative:
    from infer_utils import *

if model_name=='DA'and use_relative==False:
    da_metric_dir = os.path.join(os.path.dirname(__file__), 'other_MODELS', 'DA_metric')
    sys.path.insert(0, da_metric_dir)
    from other_MODELS.DA_metric.infer_utils import *
    use_relative = False
    dataset = 'kitti'

if model_name=='ZD':
    zd_dir = os.path.join(os.path.dirname(__file__), 'other_MODELS', 'ZoeDepth')
    sys.path.insert(0, zd_dir)
    from other_MODELS.ZoeDepth.infer_utils import *
    use_relative = False
    dataset = None # This loads the best model

if model_name=='UD':
    ud_dir = os.path.join(os.path.dirname(__file__), 'other_MODELS', 'UniDepth')
    sys.path.insert(0, ud_dir)
    from other_MODELS.UniDepth.infer_utils import *
    use_relative = False # They use one model for both KITTI and NYU

if model_name=='UN':
    ud_dir = os.path.join(os.path.dirname(__file__), 'other_MODELS', 'UniK3D')
    sys.path.insert(0, ud_dir)
    from other_MODELS.UniK3D.infer_utils import *
    use_relative = False # They use one model for both KITTI and NYU

if model_name=='AB':
    ab_dir = os.path.join(os.path.dirname(__file__), 'other_MODELS', 'AdaBins')
    sys.path.insert(0, ab_dir)
    from other_MODELS.AdaBins.infer_utils import *
    use_relative = False
    dataset = 'kitti'

print()
print("------------------")
print(f"Model   :", model_name_dict[model_name])
print(f"Dataset : KITTI")
print(f"Split   : {data_split.upper()}")
if use_relative:
    print("Type    : Relative Depth")
else:
    print("Type    : Metric Depth")
print("------------------")
print()


# -------------------------------
# Utility Functions
# -------------------------------
def load_kitti_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1216, 352))  # KITTI val cropped size
    return img

def load_kitti_depth(path):
    # Ground truth is stored as float32 PNG in KITTI depth benchmark
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = depth / 256.0  # Convert from KITTI format to meters
    return depth


def compute_delta(pred, gt, mask, thr=1.25):
    valid = mask & (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
    if not np.any(valid):
        return np.nan
    prv = pred[valid]
    gtv = gt[valid]
    ratio = np.maximum(prv/gtv, gtv/prv)
    return np.mean(ratio < thr)

def compute_abs_rel(pred, gt, mask):
    if mask is None:
        mask = np.ones_like(gt, dtype=bool)
    valid = mask & (gt > 1e-6) & np.isfinite(gt) & np.isfinite(pred)

    if np.sum(valid) == 0:
        return np.nan
    
    pred = pred[valid]
    gt = gt[valid]

    return np.mean(np.abs(gt - pred) / gt)


def align_disparity_scale_shift(pred_disp, gt_depth, mask, depth_cap):
    """
    Align disparity predictions (inverse depth) to ground truth depth using least squares.

    Args:
        pred_disp (np.ndarray): predicted disparity (inverse depth)
        gt_depth (np.ndarray): ground truth depth (in meters)
        mask (np.ndarray): valid region mask (same shape as pred_disp)
        depth_cap (float): maximum depth in meters (default: 10.0)

    Returns:
        pred_aligned_depth (np.ndarray): aligned prediction in meters
    """

    #pred_disp_norm = (pred_disp-pred_disp.min())/(pred_disp.max() - pred_disp.min())
    #gt_depth_norm = (gt_depth-gt_depth.min())/(gt_depth.max()-gt_depth.min())

    pred_disp_norm = (pred_disp)/(pred_disp.max())
    gt_depth_norm = (gt_depth)/(gt_depth.max())


    if mask is None:
        mask = (gt_depth > 1e-6) & np.isfinite(gt_depth) & np.isfinite(pred_disp)

    valid = mask & (gt_depth > 1e-6) & np.isfinite(gt_depth) & np.isfinite(pred_disp)

    if np.sum(valid) < 2:
        return 1.0 / np.clip(pred_disp, 1e-6, None)  # fallback

    #gt_disp = 1.0 / gt_depth_norm[valid]
    gt_disp = 1.0 / gt_depth[valid]

    #pred_disp_valid = pred_disp[valid]
    pred_disp_valid = pred_disp_norm[valid]


    # Construct least-squares system: target = a * pred + b
    A = np.vstack([pred_disp_valid, np.ones_like(pred_disp_valid)]).T
    b = gt_disp

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # solve for scale and shift

    scale, shift = x
    #pred_disp_aligned = scale * pred_disp + shift
    pred_disp_aligned = scale * pred_disp_norm + shift

    # Apply disparity cap (1 / max depth)
    disparity_cap = 1.0 / depth_cap
    pred_disp_aligned[pred_disp_aligned < disparity_cap] = disparity_cap

    # Convert aligned disparity back to depth
    pred_depth_aligned = 1.0 / np.clip(pred_disp_aligned, 1e-6, None)
    #pred_depth_aligned = 1/pred_disp_aligned
    pred_depth_aligned[~np.isfinite(pred_depth_aligned)] = 0.0

    return pred_depth_aligned

# -------------------------------
# Main Evaluation for KITTI
# -------------------------------
if __name__ == "__main__":

    #data_split = 'test'
    #data_split = 'val'
    #data_split = 'eigen'

    if data_split.upper() == 'VAL':
        #2011_09_26_drive_0002_sync_image_0000000005_image_02
        #2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02
        data_dir = './DATA/kitti_data/val_selection_cropped'
        image_dir = os.path.join(data_dir, 'image')
        gt_dir = os.path.join(data_dir, 'groundtruth_depth')
        img_files = sorted([os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.endswith('.png')])

    elif data_split.upper() == 'TEST':
        data_dir = './DATA/kitti_data/test_depth_prediction_anonymous'
        image_dir = os.path.join(data_dir, 'image')
        #gt_dir = os.path.join(data_dir, 'groundtruth_depth')
        save_dir = './DATA/kitti_data/test_depth_prediction_anonymous/KITTI_submission'
        os.makedirs(save_dir, exist_ok=True)
        img_files = sorted([os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.endswith('.png')])

    elif data_split.upper() == 'EIGEN':
        data_dir = './DATA/kitti_data/eigen_split/test'
        img_files = []

        # each sub‐folder is a scene, e.g. “2011_09_26_drive_0002_sync”
        for scene in sorted(os.listdir(data_dir)):
            scene_dir  = os.path.join(data_dir, scene)
            image_dir  = os.path.join(scene_dir, 'data')
            gt_dir     = os.path.join(scene_dir, 'groundtruth')
            if not os.path.isdir(image_dir) or not os.path.isdir(gt_dir):
                continue
            for fname in sorted(os.listdir(image_dir)):
                if not fname.endswith('.png'):
                    continue
                rgb_path = os.path.join(image_dir,fname)
                gt_path  = os.path.join(gt_dir,fname)
                if os.path.exists(gt_path):
                    img_files.append(rgb_path)

    else:
        raise ValueError(f"Unknown split: {data_split}")

    if dataset is None:
        model_dict = initialize_model()

    elif dataset is not None:
        model_dict = initialize_model(dataset)

    model = model_dict['MODEL']
    device = model_dict['DEVICE']
    args = model_dict['ARGS']
    transform = model_dict.get('TRANSFORM', None)

    if model is not None:
        model = model.to(device).eval()

    all_abs_rel, all_delta1 = [], []

    pbar = tqdm(img_files)

    for rgb_path in pbar:
        #rgb_path = os.path.join(image_dir, fname)

        if data_split.upper() == 'VAL':
            gt_depth_path = rgb_path.replace('image', 'groundtruth_depth',1).replace('image', 'groundtruth_depth',1)
            gt_depth = load_kitti_depth(gt_depth_path)

            if not os.path.exists(gt_depth_path):
                print(f"Missing GT for {rgb_path}")
                continue

        elif data_split.upper() == 'EIGEN':
            #gt_depth_path = rgb_path.replace('data', 'groundtruth',-1)
            head, tail = rgb_path.rsplit("data", 1)
            gt_depth_path = head + "groundtruth" + tail
            gt_depth = load_kitti_depth(gt_depth_path)

        rgb_input = load_kitti_image(rgb_path)

        # if gt_depth.shape == rgb_input.shape
        #     h, w = gt_depth.shape
        #     pred_depth = cv2.resize(
        #         pred_depth, 
        #         (w, h),                 # note: (width, height)
        #         interpolation=cv2.INTER_LINEAR
        #         )
        #print(rgb_input.shape)
        pred_depth = get_depth(rgb_input, model, device, args, transform)

        if data_split.upper() == 'TEST':
            # Save prediction as 16-bit PNG in KITTI format: depth (m) × 256
            depth_to_save = (pred_depth * 256.0).astype(np.uint16)
            save_path = os.path.join(save_dir, fname)
            cv2.imwrite(save_path, depth_to_save)

        if data_split.upper() != 'TEST':

            if pred_depth.shape != gt_depth.shape:
                # Upsample prediction to match GT
                h_gt, w_gt = gt_depth.shape
                pred_depth = cv2.resize(
                    pred_depth,
                    (w_gt, h_gt),                # note: width first, then height
                    interpolation=cv2.INTER_LINEAR)

                # Center Cropping the GT
                # h_pred, w_pred = pred_depth.shape
                # y0 = (gt_depth.shape[0] - h_pred)//2
                # x0 = (gt_depth.shape[1] - w_pred)//2
                # gt_depth = gt_depth[y0:y0+h_pred, x0:x0+w_pred]

            mask = (gt_depth > 0) & (gt_depth <= 80)

            if use_relative:
                pred_depth = align_disparity_scale_shift(pred_depth, gt_depth, mask, depth_cap=80)

            abs_rel = compute_abs_rel(pred_depth, gt_depth, mask)
            delta1 = compute_delta(pred_depth + 1e-6, gt_depth + 1e-6, mask, 1.25)

            all_abs_rel.append(abs_rel)
            all_delta1.append(delta1)

            pbar.set_postfix({'δ1': f'{np.nanmean(all_delta1):.3f}'})

    if data_split.upper() != 'TEST':
        print("\nFinal Report:")
        print(f"AbsRel: {np.nanmean(all_abs_rel):.3f}")
        print(f"δ1 Accuracy: {np.nanmean(all_delta1):.3f}")
