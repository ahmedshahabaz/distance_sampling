import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import sys

model_name_dict = {'MD': "MIDAS", 'DA': "DepthAnything", 'ZD': "ZoeDepth", 'UD': "UniDepth", 'UN': "UniK3D", 'AB': "AdaBins"}
dataset = None

if len(sys.argv) != 3:
    print("❗Usage: python eval_nyu.py <model_name> <use_relative>")
    print("    <model_name> should be one of: MD, DA, ZD, UD, AB")
    print("    <use_relative> should be: True or False (for relative depth evaluation)")
    sys.exit(1)

model_name = sys.argv[1]
use_relative_input = sys.argv[2]

# Parse boolean string to actual boolean
if use_relative_input.lower() in ['true', '1', 'yes']:
    use_relative = True
elif use_relative_input.lower() in ['false', '0', 'no']:
    use_relative = False
else:
    print("❗ Error: <use_relative> must be either 'True' or 'False'")
    sys.exit(1)

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
    dataset = 'nyu'

if model_name=='ZD':
    zd_dir = os.path.join(os.path.dirname(__file__), 'other_MODELS', 'ZoeDepth')
    sys.path.insert(0, zd_dir)
    from other_MODELS.ZoeDepth import hf_midas_patch
    from other_MODELS.ZoeDepth.infer_utils import *
    use_relative = False
    dataset = None # This loads the best model

if model_name=='UD':
    ud_dir = os.path.join(os.path.dirname(__file__), 'other_MODELS', 'UniDepth')
    sys.path.insert(0, ud_dir)
    from other_MODELS.UniDepth.infer_utils import *
    use_relative = False
    dataset = None # They use one model for both KITTI and NYU

if model_name=='UN':
    ud_dir = os.path.join(os.path.dirname(__file__), 'other_MODELS', 'UniK3D')
    sys.path.insert(0, ud_dir)
    from other_MODELS.UniK3D.infer_utils import *
    use_relative = False
    dataset = None # They use one model for both KITTI and NYU

if model_name=='AB':
    ab_dir = os.path.join(os.path.dirname(__file__), 'other_MODELS', 'AdaBins')
    sys.path.insert(0, ab_dir)
    from other_MODELS.AdaBins.infer_utils import *
    use_relative = False
    dataset = 'nyu'

print()
print("------------------")
print("Model   :", model_name_dict[model_name])
print("Dataset : NYUv2")
if use_relative:
    print("Type    : Relative Depth")
else:
    print("Type    : Metric Depth")
print("------------------")
print()

# -------------------------------
# Utility Functions
# -------------------------------
def load_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))  # Adjust if needed
    return img

def load_depth(filepath):
    depth = cv2.imread(filepath, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = cv2.resize(depth, (640, 480))  # Match RGB size
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

    pred_disp_norm = (pred_disp)/(pred_disp.max())


    if mask is None:
        mask = (gt_depth > 1e-6) & np.isfinite(gt_depth) & np.isfinite(pred_disp)

    valid = mask & (gt_depth > 1e-6) & np.isfinite(gt_depth) & np.isfinite(pred_disp)

    if np.sum(valid) < 2:
        return 1.0 / np.clip(pred_disp, 1e-6, None)  # fallback

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
# Main Evaluation Loop
# -------------------------------
if __name__ == "__main__":

    data_dir = './DATA/nyu_data/nyu2_test'
    rgb_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_colors.png')])

    #data_dir = './DATA/nyu_data/nyu2_train'
    #rgb_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True))
    
    if dataset is None:
        model_dict = initialize_model()

    elif dataset is not None:
        model_dict = initialize_model(dataset)

    model = model_dict['MODEL']
    device = model_dict['DEVICE']
    args = model_dict['ARGS']
    transform = model_dict.get('TRANSFORM', None)
    net_w = model_dict.get('net_w', None)
    net_h = model_dict.get('net_h', None)

    if model is not None:
        model = model.to(device).eval()

    all_abs_rel, all_delta1 = [], []

    pbar = tqdm(rgb_files)

    for f in pbar:
        base_name = f.replace('_colors.png', '')
        rgb_path = os.path.join(data_dir, f)
        depth_path = os.path.join(data_dir, base_name + '_depth.png')


        if not os.path.exists(depth_path):
            print(f"Missing GT for {rgb_path}")
            continue


        # Load inputs
        rgb_input = load_image(rgb_path)
        gt_depth = load_depth(depth_path)

        pred_depth = get_depth(rgb_input,model,device,args,transform)

        # Create valid mask
        mask = (gt_depth > 0) & (gt_depth < 10000)

        if use_relative:
            pred_depth = align_disparity_scale_shift(pred_depth, gt_depth, mask, depth_cap=10000.0)

        else:
            pred_depth = pred_depth * 1000.0

        abs_rel = compute_abs_rel(pred_depth, gt_depth, mask)
        delta1 = compute_delta(pred_depth + 1e-6, gt_depth + 1e-6, mask, 1.25)

        all_abs_rel.append(abs_rel)
        all_delta1.append(delta1)

        pbar.set_postfix({'δ1': f'{np.nanmean(all_delta1):.3f}'})

    # -------------------------------
    # Final Report
    # -------------------------------
    print("\nFinal Report:")
    print(f"AbsRel: {np.nanmean(all_abs_rel):.3f}")
    print(f"δ1 Accuracy: {np.nanmean(all_delta1):.3f}")




