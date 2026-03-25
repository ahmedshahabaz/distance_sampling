import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from utils.pipeline_utils import resolve_pred_dir
from utils.script_utils import parse_depth_cli, print_run_banner

model_name, use_relative, use_calb, root_dir, _ = parse_depth_cli(
    "Print filtered/unfiltered distance error summary from eval_ref_point_distances output."
)
data_dir = resolve_pred_dir(model_name, use_relative, root_dir)
print_run_banner(model_name, use_relative, use_calb, data_dir)

data_file = "ref_point_stats_calb.json"

'''
Applies only for Metric Depth models, where we have the option to calibrate predictions.
If using calibrated predictions, we want to use the CALB version of the data file,
which contains the relevant error metrics for the calibrated predictions.
'''
if not use_relative and not use_calb:
    data_file = "ref_point_stats.json"

with open(os.path.join(data_dir, data_file), "r") as f:
    data = json.load(f)



point_selects = ["mean", "median", "all"]#, "outliers_rmv"]#, "smooth"]

#error_metrics = ["diff_err","abs_err", "abs_rel", "rmse", "delta1", "delta2", "delta3"]
error_metrics = ["abs_err", "abs_rel", "rmse", "delta1", "delta2", "delta3"]

not_percent = ["abs_err", "rmse"]

summary_data_filtered = []
summary_data_unfiltered = []

for camera_type, video in data.items():
    for vid_id, vid_data in video.items():
        for line in vid_data["lines"]:
            gt_depth = line["GT"]
            for p_slct in point_selects:
                if p_slct in line["errors"]:
                    for metric in error_metrics:
                        err_val = line["errors"][p_slct][metric]
                        if err_val == -1:
                            continue
                        if metric not in not_percent:
                            err_val = err_val
                        row = {
                            "Camera Type": camera_type,
                            "Scene ID": vid_id,
                            "Point Select": p_slct,
                            "Metric": metric.replace("_", "\\_"),
                            "Value": err_val
                        }
                        # Add to unfiltered always
                        summary_data_unfiltered.append(row.copy())
                        # Add to filtered only if gt_depth is valid
                        if gt_depth is not None and gt_depth <= 15:
                            summary_data_filtered.append(row.copy())

# Convert to DataFrames
df_filt = pd.DataFrame(summary_data_filtered).dropna()
df_unfilt = pd.DataFrame(summary_data_unfiltered).dropna()



# --- robust stats helpers ---
def mad_scaled(x):
    """Scaled MAD (robust sigma): 1.4826 * median(|x - median(x)|)."""
    x = np.asarray(x)
    m = np.nanmedian(x)
    mad0 = np.nanmedian(np.abs(x - m))
    return 1.4826 * mad0

def iqr(x):
    """Interquartile Range: Q3 - Q1."""
    x = np.asarray(x)
    q1 = np.nanquantile(x, 0.25)
    q3 = np.nanquantile(x, 0.75)
    return float(q3 - q1)

# --- summary maker (replaces your make_summary) ---
def make_summary(df, round_to=3):
    out = df.groupby(["Point Select", "Metric"])["Value"].agg(
        Median="median",
        MAD=mad_scaled,       # scaled MAD so it's comparable to SD
        IQR=iqr               # works nicely for long-tailed / bounded metrics
    ).reset_index()
    return out.round(round_to)

summary_filt = make_summary(df_filt).round(3)
summary_unfilt = make_summary(df_unfilt).round(3)


merged = summary_unfilt.merge(
    summary_filt,
    on=["Point Select", "Metric"],
    suffixes=("_all", "_≤15m")
)
print(merged.to_string(index=False))