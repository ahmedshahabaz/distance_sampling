"""
analysis_utils.py – Plotting and analysis utilities.

Provides:
  - Pixel-wise variance / VMR computation over a frame stack
  - Variance-difference and VMR-difference map plots
  - Per-bbox temporal depth tracking (update_bbox_results)
  - Temporal variation plots (plot_temporal_variation)
"""

import gc
import os

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cv2 import ellipse2Poly

from utils.calb_utils import get_depth_on_circle


# ── Constants ─────────────────────────────────────────────────────────────────

FONT_SIZE = 19
# Fixed symmetric range for variance-difference colour maps.
VAR_DIFF_VMAX = 0.010


# ── Finite-value helpers ───────────────────────────────────────────────────────

def _finite_values(arr, max_samples=500_000):
    """
    Return finite values from arr, subsampled to at most max_samples elements.
    Avoids loading multi-million-element arrays into histogram bins in full.
    """
    vals = np.ravel(arr)
    if vals.size > max_samples:
        step = int(np.ceil(vals.size / max_samples))
        vals = vals[::step]
    return vals[np.isfinite(vals)]


def _safe_percentile(arr, q, default=1.0):
    vals = _finite_values(arr)
    if vals.size == 0:
        return default
    out = np.percentile(vals, q)
    return out if np.isfinite(out) else default


def _safe_absmax(arr, default=1.0):
    vals = _finite_values(arr)
    if vals.size == 0:
        return default
    out = np.max(np.abs(vals))
    return out if (np.isfinite(out) and out != 0) else default


def _safe_stats(arr):
    """Return {mean, median, max} for finite values in arr."""
    vals = _finite_values(arr)
    if vals.size == 0:
        return {"mean": np.nan, "median": np.nan, "max": np.nan}
    return {
        "mean":   np.mean(vals),
        "median": np.median(vals),
        "max":    np.max(vals),
    }


# ── File-type helpers ──────────────────────────────────────────────────────────

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

def is_video_file(filename):
    return filename.lower().endswith(VIDEO_EXTS)

def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTS)


# ── Pixel-wise variance computation ───────────────────────────────────────────

def _compute_pixelwise_var_mean(stack):
    """
    Compute per-pixel mean and variance over a frame stack using only finite
    values (NaN / Inf frames are skipped per pixel).

    Uses a running sum / sum-of-squares accumulator to keep memory constant
    regardless of the number of frames.

    Parameters
    ----------
    stack : (T, H, W) float array

    Returns
    -------
    var_map  : (H, W) float64 – per-pixel variance (NaN where no valid frames)
    mean_map : (H, W) float64 – per-pixel mean
    """
    h, w = stack.shape[1], stack.shape[2]
    sum_map   = np.zeros((h, w), dtype=np.float64)
    sumsq_map = np.zeros((h, w), dtype=np.float64)
    count_map = np.zeros((h, w), dtype=np.int32)

    for i in range(stack.shape[0]):
        frame = np.asarray(stack[i], dtype=np.float64)
        if frame.shape != (h, w):
            raise ValueError(
                f"Inconsistent frame shape at index {i}: "
                f"got {frame.shape}, expected {(h, w)}"
            )
        valid_idx = np.flatnonzero(np.isfinite(frame.ravel()))
        if valid_idx.size == 0:
            continue
        vals = frame.ravel()[valid_idx]
        sum_map.ravel()[valid_idx]   += vals
        sumsq_map.ravel()[valid_idx] += vals * vals
        count_map.ravel()[valid_idx] += 1

    mean_map = np.full((h, w), np.nan, dtype=np.float64)
    var_map  = np.full((h, w), np.nan, dtype=np.float64)
    valid    = count_map > 0

    mean_map[valid] = sum_map[valid] / count_map[valid]
    # Var = E[X²] - E[X]²; clamp to 0 to avoid tiny negatives from float rounding.
    var_vals = sumsq_map[valid] / count_map[valid] - mean_map[valid] ** 2
    var_map[valid] = np.maximum(var_vals, 0.0)

    return var_map, mean_map


# ── Colorbar helper ────────────────────────────────────────────────────────────

def _save_vertical_colorbar(save_path, vmin, vmax, cmap='bwr',
                             label='Change in Variance'):
    """Save a standalone vertical colorbar figure (used as a legend image)."""
    fig, cax = plt.subplots(figsize=(5, 7))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm   = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.set_label(label, size=FONT_SIZE, rotation=270, labelpad=26)
    cbar.ax.tick_params(labelsize=FONT_SIZE, pad=8)

    # Small axis padding so tick labels don't sit on figure edges.
    span      = float(vmax - vmin)
    axis_pad  = max(span * 0.03, 1e-8)
    cbar.ax.set_ylim(vmin - axis_pad, vmax + axis_pad)

    fig.subplots_adjust(left=0.25, right=0.82, top=0.98, bottom=0.06)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Per-bbox result accumulator ────────────────────────────────────────────────

def update_bbox_results(bbox_results, circle_bbox, scale_x, scale_y, scale_r,
                        pred_disparity, calb_disparity, vid_file_name, frame_idx):
    """
    Accumulate mean predicted and calibrated disparity on each circle patch.

    Called once per frame; appends values to bbox_results[patch_idx]['pred']
    and ['calb'] for later temporal plotting.

    Parameters
    ----------
    bbox_results   : dict  – {patch_idx: {pred, calb, video, frame}}
    circle_bbox    : list  – annotation list for the site
    pred_disparity : H×W array  – normalised predicted disparity
    calb_disparity : H×W array  – calibrated disparity
    """
    for bbox_idx, bbox_info in enumerate(circle_bbox):
        cx = int(bbox_info['x'] * scale_x)
        cy = int(bbox_info['y'] * scale_y)
        r  = int(bbox_info['radius'] * scale_r)
        circle_pts = ellipse2Poly((cx, cy), (r, r), 0, 0, 360, delta=5)
        if len(circle_pts) < 3:
            continue

        pred_vals = get_depth_on_circle(pred_disparity, circle_pts)
        calb_vals = get_depth_on_circle(calb_disparity, circle_pts)

        bbox_results[bbox_idx]['pred'].append(np.mean(pred_vals))
        bbox_results[bbox_idx]['calb'].append(np.mean(calb_vals))
        bbox_results[bbox_idx]['video'].append(vid_file_name)
        bbox_results[bbox_idx]['frame'].append(frame_idx)


# ── Variance / VMR plot functions ──────────────────────────────────────────────

def _plot_variance_difference(var_pred, var_calb, save_path):
    """
    Save a pixel-wise (var_pred − var_calb) difference map with a shared
    symmetric colour scale.  A standalone colorbar legend is saved once
    alongside the first map written to that directory.
    """
    diff = np.full(var_pred.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(var_pred) & np.isfinite(var_calb)
    diff[valid] = var_pred[valid] - var_calb[valid]

    vmax = VAR_DIFF_VMAX
    plt.figure(figsize=(11, 7))
    plt.imshow(np.ma.masked_invalid(diff), cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Write the colorbar legend once per output directory.
    colorbar_path = os.path.join(os.path.dirname(save_path), "var_diff_colorbar.jpeg")
    if not os.path.exists(colorbar_path):
        _save_vertical_colorbar(colorbar_path, -vmax, vmax,
                                cmap='bwr', label='Change in Variance')


def plot_depth_variance(depth_stack_pred, depth_stack_calb,save_path_prefix):
    """
    Compute and save variance / VMR comparison plots for one video.

    Generates three output files:
      <prefix>_var_diff.jpeg  – pixel-wise variance difference (pred − calb)
      <prefix>_variance.jpeg  – side-by-side variance maps
      <prefix>_vmr.png        – side-by-side VMR maps
      <prefix>_vmr_diff.jpeg  – pixel-wise VMR difference

    Parameters
    ----------
    depth_stack_pred : (T, H, W) float32 – normalised predicted depth per frame
    depth_stack_calb : (T, H, W) float32 – calibrated depth per frame
    save_path_prefix : str  – base filename (without extension)
    """
    var_pred, mean_pred = _compute_pixelwise_var_mean(depth_stack_pred)
    var_calb, mean_calb = _compute_pixelwise_var_mean(depth_stack_calb)

    # Free the large stacks before allocating more arrays.
    del depth_stack_pred, depth_stack_calb
    gc.collect()

    # Variance-to-Mean Ratio (VMR): var / mean, guarded against division by zero.
    eps = 1e-8
    vmr_pred = np.divide(
        var_pred, mean_pred + eps,
        out=np.full_like(var_pred, np.nan),
        where=np.isfinite(var_pred) & np.isfinite(mean_pred) & (np.abs(mean_pred) > eps),
    )
    vmr_calb = np.divide(
        var_calb, mean_calb + eps,
        out=np.full_like(var_calb, np.nan),
        where=np.isfinite(var_calb) & np.isfinite(mean_calb) & (np.abs(mean_calb) > eps),
    )
    vmr_diff = np.full(vmr_pred.shape, np.nan, dtype=np.float64)
    valid_vmr = np.isfinite(vmr_pred) & np.isfinite(vmr_calb)
    vmr_diff[valid_vmr] = vmr_pred[valid_vmr] - vmr_calb[valid_vmr]

    stats_var_pred = _safe_stats(var_pred)
    stats_var_calb = _safe_stats(var_calb)
    stats_vmr_pred = _safe_stats(vmr_pred)
    stats_vmr_calb = _safe_stats(vmr_calb)

    # ── Figure 1: Variance difference map ────────────────────────────────────
    _plot_variance_difference(var_pred, var_calb,
                               save_path_prefix + '_var_diff.jpeg')

    # ── Figure 2: Side-by-side variance maps ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    for ax, var_map, stats, title in [
        (axes[0], var_pred, stats_var_pred, 'Variance (Predicted)'),
        (axes[1], var_calb, stats_var_calb, 'Variance (Calibrated)'),
    ]:
        ax.set_title(title, fontsize=30)
        ax.imshow(var_map, cmap='bwr', vmin=0, vmax=_safe_percentile(var_map, 99))
        ax.axis('off')
        ax.text(
            0.02, 0.95,
            f"Mean: {stats['mean']:.4f}\n"
            f"Median: {stats['median']:.4f}\n"
            f"Max: {stats['max']:.4f}",
            transform=ax.transAxes, fontsize=FONT_SIZE, color='white',
            va='top', ha='left',
            bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'),
        )
    plt.savefig(save_path_prefix + '_variance.jpeg', dpi=300)
    plt.close()

    # ── Figure 3: Side-by-side VMR maps ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    for ax, vmr_map, stats, title in [
        (axes[0], vmr_pred, stats_vmr_pred, 'Variance-to-Mean Ratio (Predicted)'),
        (axes[1], vmr_calb, stats_vmr_calb, 'Variance-to-Mean Ratio (Calibrated)'),
    ]:
        ax.set_title(title, fontsize=FONT_SIZE)
        im = ax.imshow(vmr_map, cmap='plasma', vmin=0,
                       vmax=_safe_percentile(vmr_map, 99))
        ax.axis('off')
        plt.colorbar(im, ax=ax).ax.tick_params(labelsize=FONT_SIZE)
        ax.text(
            0.02, 0.95,
            f"Mean: {stats['mean']:.4f}\n"
            f"Median: {stats['median']:.4f}\n"
            f"Max: {stats['max']:.4f}",
            transform=ax.transAxes, fontsize=FONT_SIZE, color='white',
            va='top', ha='left',
            bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'),
        )
    plt.savefig(save_path_prefix + '_vmr.png', dpi=300)
    plt.close()

    # ── Figure 4: VMR difference map ─────────────────────────────────────────
    vmax = _safe_absmax(vmr_diff)
    plt.figure(figsize=(11, 7))
    plt.imshow(np.where(np.isfinite(vmr_diff), vmr_diff, np.nan),
               cmap='bwr', vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar()
    cbar.set_label('VMR Change', size=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    plt.savefig(save_path_prefix + '_vmr_diff.jpeg', dpi=300)
    plt.close()


# ── Temporal depth variation plot ─────────────────────────────────────────────

def plot_temporal_variation(plot_dir, bbox_results, site_number,
                            frames_per_video=300):
    """
    Plot predicted vs calibrated disparity over time for each circle patch.

    Parameters
    ----------
    bbox_results     : dict  – {patch_idx: {pred, calb, video, frame}}
    site_number      : str   – site label for the plot title
    frames_per_video : int   – used to space x-axis tick marks
    """
    for bbox_idx, data in bbox_results.items():
        n_frames = len(data['pred'])
        step     = 3 * frames_per_video

        plt.figure(figsize=(20, 16))
        plt.plot(data['pred'], label='Predicted')
        plt.plot(data['calb'], label='Calibrated')
        plt.title(f"Site {site_number}, Circle #{bbox_idx} – Temporal Depth",
                  fontsize=FONT_SIZE)
        plt.xlabel("Frame index (all videos concatenated)", fontsize=FONT_SIZE)
        plt.ylabel("Mean depth (normalised)", fontsize=FONT_SIZE)

        xticks = np.arange(0, n_frames + 1, step)
        plt.xticks(xticks, [str(x) for x in xticks], fontsize=FONT_SIZE)
        plt.yticks(fontsize=FONT_SIZE)
        plt.legend(fontsize=FONT_SIZE)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}_circle_{bbox_idx}_temporal_depth.jpeg", dpi=300)
        plt.close()
