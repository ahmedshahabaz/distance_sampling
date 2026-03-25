import json
import os
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import f_oneway, kruskal, levene, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from utils.pipeline_utils import resolve_pred_dir
from utils.script_utils import parse_depth_cli, print_run_banner

FONT_SIZE = 15
TOD_ORDER = ["DAWN", "DAY", "DUSK", "NIGHT"]
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "legend.title_fontsize": FONT_SIZE,
})

model_name, use_relative, use_calb, root_dir, annot_dir = parse_depth_cli(
    "Plot and summarise depth error statistics by time of day."
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
    json_data = json.load(f)


TOD_df = pd.read_csv(os.path.join(annot_dir, 'TOD.csv'))
video_to_TOD = dict(zip(TOD_df['VIDEO'], TOD_df['TOD']))

#error_metrics = ["abs_err", "abs_rel", "rmse", "delta1", "delta2", "delta3"]

error_metrics = ["diff_err","abs_err", "abs_rel", "rmse", "delta1"]
not_percent = ["diff_err","abs_err", "rmse"]

# Extract metric data with associated TOD
point_select = 'median'
metric = 'diff_err' # it is ME
#metric = 'abs_rel'

box_plot_data = []
site_data = []

for cam_id, videos in json_data.items():
    for vid_key, vid_info in videos.items():
        file_name = vid_info['file_name']
        base_name = os.path.splitext(file_name)[0]

        tod = video_to_TOD.get(base_name)
        if tod is None:
            continue  # skip if no TOD info

        for line in vid_info['lines']:
            GT_distance =  line.get("GT", {})
            err_val = line.get("errors", {}).get(point_select, {}).get(metric, None)
            if err_val is not None and err_val != -1:
                box_plot_data.append({
                    'site': cam_id,
                    'video': base_name,
                    'tod': tod,
                    'distance': GT_distance,
                    'value': err_val
                })


box_plot_df = pd.DataFrame(box_plot_data)
box_plot_df['distance'] = pd.to_numeric(box_plot_df['distance'], errors='coerce')
box_plot_df['value'] = pd.to_numeric(box_plot_df['value'], errors='coerce')
box_plot_df = box_plot_df.dropna(subset=['distance', 'value'])

def mad_scaled(x):
    """Scaled MAD: 1.4826 * median(|x - median(x)|)."""
    x = np.asarray(x)
    m = np.nanmedian(x)
    mad0 = np.nanmedian(np.abs(x - m))
    return 1.4826 * mad0

def iqr(x):
    """Interquartile Range = Q3 - Q1."""
    x = np.asarray(x)
    return float(np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25))

rows = []
for m in error_metrics:
    metric_data = []
    for cam_id, videos in json_data.items():
        for vid_key, vid_info in videos.items():
            file_name = vid_info['file_name']
            base_name = os.path.splitext(file_name)[0]
            tod = video_to_TOD.get(base_name)
            if tod is None:
                continue
            for line in vid_info['lines']:
                val = line.get("errors", {}).get(point_select, {}).get(m, None)
                if val is not None and val != -1:
                    metric_data.append({'tod': tod, 'metric': m, 'video': base_name, 'value': val})
    df = pd.DataFrame(metric_data)
    #grouped = df.groupby(['tod', 'metric'])['value'].agg(['mean','median','std','min','max','count']).reset_index()
    # robust stats by ToD
    grouped = df.groupby(['tod', 'metric']).agg(
        Median=('value', 'median'),
        MAD=('value', mad_scaled),
        IQR=('value', iqr),
        Count=('value', 'count'),
        Videos=('video', 'nunique')
    ).reset_index()
    rows.append(grouped)

summary_df = pd.concat(rows, ignore_index=True)
#summary_df = summary_df[['tod', 'metric', 'mean', 'median', 'std', 'min', 'max', 'count']]
summary_df = summary_df[['tod', 'metric', 'Median', 'MAD', 'IQR', 'Count', 'Videos']]
summary_df = summary_df.sort_values(['tod', 'metric'])
print(summary_df.to_string(index=False))

filtered_df = box_plot_df[box_plot_df['distance'] <= 15]

vid_counts = (
    filtered_df.loc[:, ['tod','video']]
    .drop_duplicates()                # one row per (tod, video)
    .groupby('tod')['video']
    .nunique()
)


plt.figure(figsize=(10,8))
ax = sns.boxplot(
    data=filtered_df, 
    x='tod', 
    y='value',
    order=TOD_ORDER,
    showfliers=False,
    showmeans=True,
    palette=None,  # No color fill
    boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':2},
    medianprops={'color':'orange', 'linewidth':1},
    meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black"}
)

sns.despine()
plt.xlabel("Time of Day", fontsize=FONT_SIZE)
plt.ylabel("Error (m)", fontsize=FONT_SIZE)
plt.grid(axis='y', linestyle=':', linewidth=1.2, color='gray', alpha=0.5)

# Add rectangle around the plotting area
ax.add_patch(
    patches.Rectangle(
        (ax.get_xlim()[0], ax.get_ylim()[0]),
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        fill=False,
        edgecolor="black",
        linewidth=1.5,
        zorder=10
    )
)

labels = [t.get_text() for t in ax.get_xticklabels()]

ymin, ymax = ax.get_ylim()
pad = 0.06 * (ymax - ymin)
ax.set_ylim(ymin, ymax + pad)
for i, lab in enumerate(labels):
    ax.text(i, ymax + 0.02*(ymax - ymin), f"n={vid_counts.get(lab,0)}",
            ha='center', va='bottom', fontsize=FONT_SIZE)

plt.tight_layout()
#plt.savefig("REPORTS/boxplot_by_ToD_upto15m.svg", format="svg", bbox_inches="tight")


# Box plot by site (only boxes, filtered)
plt.figure(figsize=(15,10))
ax = sns.boxplot(
    data=filtered_df, 
    x='site', 
    y='value',
    showfliers=False,
    showmeans=True,
    palette=None,  # No color palette
    boxprops={'facecolor':'none', 'edgecolor':'black', 'linewidth':2},
    medianprops={'color':'orange', 'linewidth':1},
    meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black"}
)

sns.despine()
plt.xlabel("Site", fontsize=FONT_SIZE)
plt.ylabel('Error (m)', fontsize=FONT_SIZE)
plt.grid(axis='y', linestyle=':', linewidth=1.2, color='gray', alpha=0.5)
plt.xticks(rotation=45, ha='right',fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)

# Add rectangle patch around the plot area
ax.add_patch(
    patches.Rectangle(
        # Position & size from current axis limits
        (ax.get_xlim()[0], ax.get_ylim()[0]),
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        fill=False,
        edgecolor="black",
        linewidth=2,
        zorder=10
    )
)

plt.tight_layout()
#plt.savefig("REPORTS/boxplot_by_site_upto15m.svg", format="svg", bbox_inches="tight")
plt.show()
