import json
import os

import numpy as np
import pandas as pd

from utils.pipeline_utils import resolve_pred_dir
from utils.script_utils import get_ref_point_circles, parse_depth_cli, print_run_banner

model_name, use_relative, use_calb, root_dir, _ = parse_depth_cli(
    "Print per-distance-bin error summary from eval_ref_point_distances output."
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


# Define distance bins (adjust based on your dataset range)
distance_bins = np.arange(0, 27, 3)  # Example: Bins from 0 to 100m in steps of 5m
bin_labels = [f"{distance_bins[i]}-{distance_bins[i+1]}m" for i in range(len(distance_bins)-1)]

pred_point_sample = "median"

print("Point sampling method: ", pred_point_sample.upper())
print()

# Extract relevant data for outliers_rmv calibration
filtered_data = []
for camera_type, video in data.items():
    for vid_id, vid_data in video.items():
        for circle in get_ref_point_circles(vid_data):
            if pred_point_sample in circle["errors"]:
                gt_depth = circle["GT"]
                pred_depth = circle[f"calb_depth_{pred_point_sample}"]
                #diff_error = circle["errors"][pred_point_sample]["diff_err"]
                abs_error = circle["errors"][pred_point_sample]["abs_err"]
                abs_rel_error = circle["errors"][pred_point_sample]["abs_rel"] #* 100  # Convert to percentage
                sq_rel_error = circle["errors"][pred_point_sample]["sq_rel"] #* 100  # Convert to percentage
                rmse_error = circle["errors"][pred_point_sample]["rmse"]
                delta1_acc = circle["errors"][pred_point_sample]["delta1"]

                filtered_data.append({
                    "GT Depth": gt_depth,
                    "Predicted Depth": pred_depth,
                    #"ME":diff_error,
                    "Abs Error":abs_error,
                    "Abs Rel Error": abs_rel_error,
                    "Sq Rel Error": sq_rel_error,
                    "RMSE": rmse_error,
                    "Delta1": delta1_acc
                })

# Convert to DataFrame
df_filtered = pd.DataFrame(filtered_data)
df_filtered = df_filtered.dropna()

# Assign bins
df_filtered["Distance Bin"] = pd.cut(df_filtered["GT Depth"], bins=distance_bins, labels=bin_labels, right=True)

# Compute mean values per bin
summary_table = df_filtered.groupby("Distance Bin", observed=False).agg(
    #Mean_Pred_Depth=("Predicted Depth", "mean"),
    #Mean_GT_Depth=("GT Depth", "mean"),
    #ME=("ME", "mean"),
    MAE=("Abs Error", "mean"),
    Mn_AbsRel=("Abs Rel Error", "mean"),
    #Mean_Sq_Rel_Err=("Sq Rel Err", "mean"),
    Mn_Delta1 = ("Delta1", "mean"),
    RMSE=("RMSE", "mean")
).reset_index()

# Format Percentage Columns
#summary_table["Mean_Abs_Rel_Error"] = summary_table["Mean_Abs_Rel_Error"].apply(lambda x: f"{x:.2f}%")
#summary_table["ME"] = summary_table["ME"].apply(lambda x: f"{x:.2f}")
summary_table["MAE"] = summary_table["MAE"].apply(lambda x: f"{x:.2f}")
summary_table["Mn_AbsRel"] = summary_table["Mn_AbsRel"].apply(lambda x: f"{x:.2f}")
summary_table["Mn_Delta1"] = summary_table["Mn_Delta1"].apply(lambda x: f"{x:.2f}")
summary_table["RMSE"] = summary_table["RMSE"].apply(lambda x: f"{x:.2f}")

print(summary_table)
df_15 = df_filtered[df_filtered["GT Depth"] <= 15]
print(f"\nOverall Mean AbsRel (<=15m): {df_15['Abs Rel Error'].mean():.3f}")
print(f"Overall Mean Delta1 (<=15m): {df_15['Delta1'].mean():.3f}")
