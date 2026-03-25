import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# --- shared base style (font overridden per-plot) ---
# text.usetex=True delegates rendering to LaTeX, matching elsarticle's
# Computer Modern font exactly. Requires a LaTeX installation (texlive etc.).
_BASE_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{lmodern}\renewcommand{\seriesdefault}{bx}\normalfont",
    "font.weight": "bold",
    "axes.labelweight": "bold",
}

# --- shared paths ---
_DATA = Path("./DATA")
xlsx_audit_dpt = _DATA / "animal_dist_rslt_AUDIT_DPT_full.xlsx"
xlsx_audit_da  = _DATA / "animal_dist_rslt_AUDIT_DA_full.xlsx"
xlsx_patch     = _DATA / "animal_dist_rslt_DA_full.xlsx"

# --- distance bins ---
distance_bins = [0, 3, 6, 9, 12, 15, 18]
bin_labels    = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18"]

# --- colors ---
_cmap = cm.get_cmap("bwr")
COLORS = {
    "Ours (Patch-Calibrated)": _cmap(0.0),
    "AUDIT-DA":                _cmap(0.43),
    "AUDIT-DPT":               _cmap(1.0),
}

DATASETS = [
    ("Ours (Patch-Calibrated)", xlsx_patch),
    ("AUDIT-DA",                xlsx_audit_da),
    ("AUDIT-DPT",               xlsx_audit_dpt),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_clean(path, gt_col="GT Dist", pred_col="Mask Dist"):
    df = pd.read_excel(path)
    if "Site" in df.columns:
        df = df[~df["Site"].astype(str).str.upper().isin(["MEAN", "MEDIAN"])]
    if not {gt_col, pred_col}.issubset(df.columns):
        raise ValueError(f"{path.name} missing required columns.")
    df = df[[gt_col, pred_col]].rename(columns={gt_col: "GT", pred_col: "Pred"})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df[(df["GT"] > 0) & (df["Pred"] > 0)]


def load_for_bins(path, gt_col="GT Dist", pred_col="Mask Dist", err_col="abs_rel"):
    """Load a sheet and return a clean DF with GT, abs_rel, and Distance Bin."""
    df = pd.read_excel(path)
    if "Site" in df.columns:
        df = df[~df["Site"].astype(str).str.upper().isin(["MEAN", "MEDIAN", "MAD"])]
    if gt_col not in df.columns:
        raise ValueError(f"{path.name} missing '{gt_col}'.")

    if err_col in df.columns:
        df = df[[gt_col, err_col]].rename(columns={gt_col: "GT", err_col: "abs_rel"})
    else:
        if pred_col not in df.columns:
            raise ValueError(f"{path.name} missing '{err_col}' and '{pred_col}'.")
        df = df[[gt_col, pred_col]].copy()
        df.columns = ["GT", "Pred"]
        df["abs_rel"] = np.abs(df["Pred"] - df["GT"]) / df["GT"]
        df = df[["GT", "abs_rel"]]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["Distance Bin"] = pd.cut(df["GT"], bins=distance_bins, labels=bin_labels, right=True)
    return df


def median_abs_dev(x, scale=True):
    med  = np.median(x)
    mad0 = np.median(np.abs(x - med))
    return mad0 * 1.4826 if scale else mad0


# ── Plot 1: KDE of GT vs predicted distances ─────────────────────────────────

plt.rcParams.update({**_BASE_RC, "font.size": 25, "axes.labelsize": 25,
                     "legend.fontsize": 25, "legend.title_fontsize": 25})

df_patch = load_clean(xlsx_patch)
df_da    = load_clean(xlsx_audit_da)
df_dpt   = load_clean(xlsx_audit_dpt)
gt_all   = pd.concat([df_patch["GT"], df_da["GT"], df_dpt["GT"]], ignore_index=True)

print("Shapes:", df_patch.shape, df_da.shape, df_dpt.shape)

plt.figure(figsize=(10, 7))
sns.kdeplot(df_patch["Pred"], label="Ours (Patch-Calibrated)", clip=(0, None))
sns.kdeplot(df_da["Pred"],    label="AUDIT--DA",               clip=(0, None))
sns.kdeplot(df_dpt["Pred"],   label="AUDIT--DPT",              clip=(0, None))
sns.kdeplot(gt_all,           label="Ground Truth",            clip=(0, None),
            linestyle="--", color="red")

plt.xlabel("Distance (m)")
plt.ylabel("Density")
plt.legend(title="Method", frameon=False)
plt.tight_layout()

out_kde = Path("./REPORTS/KDE_gt_vs_pred_audit_ours.svg")
plt.savefig(out_kde, dpi=300)
plt.close()
print(f"Saved KDE plot → {out_kde}")


# ── Plot 2: grouped bar chart by distance bin ─────────────────────────────────

plt.rcParams.update({**_BASE_RC, "font.size": 15, "axes.labelsize": 15,
                     "legend.fontsize": 15, "legend.title_fontsize": 15})

stats_list    = []
overall_lines = []

for label, path in DATASETS:
    df = load_for_bins(path)

    bin_stats = df.groupby("Distance Bin")["abs_rel"].agg(
        count="count", median="median", mad=median_abs_dev
    ).reset_index()
    bin_stats["Method"] = label
    stats_list.append(bin_stats)

    overall_med = float(np.median(df["abs_rel"])) if not df.empty else float("nan")
    overall_mad = float(median_abs_dev(df["abs_rel"])) if not df.empty else float("nan")
    overall_lines.append(f"{label}: med={overall_med:.3f}, MAD={overall_mad:.3f}")

all_stats   = pd.concat(stats_list, ignore_index=True)
median_piv  = all_stats.pivot(index="Distance Bin", columns="Method", values="median").reindex(bin_labels)
methods     = [m for m, _ in DATASETS]
n_bins, n_m = len(bin_labels), len(methods)
x           = np.arange(n_bins)
width       = 0.8 / n_m

fig, ax = plt.subplots(figsize=(12, 8))

for i, m in enumerate(methods):
    ax.bar(
        x + i * width - (n_m - 1) * width / 2,
        median_piv[m].values,
        width=width,
        label=m,
        edgecolor="black",
        linewidth=0.5,
        color=COLORS.get(m),
    )

ax.set_xlabel(r"Distance Bin (m)")
ax.set_ylabel(r"Median Absolute Relative Error")
ax.set_xticks(x)
ax.set_xticklabels(bin_labels)
ax.legend(title=r"Method", frameon=False, loc="upper center")

caption = r"\begin{{tabular}}{{c}} {} \\ {} \end{{tabular}}".format(
    overall_lines[0], r" \textbar\ ".join(overall_lines[1:])
)
fig.suptitle(caption, x=0.5, y=0.85, ha="center", va="bottom",
             fontsize=17, style="italic", linespacing=1.5, multialignment="center")

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_bar = Path("./REPORTS/animal_dist_bin_ALL_METHODS.svg")
plt.savefig(out_bar, format="svg", bbox_inches="tight")
plt.close()
print(f"Saved grouped distance-bin plot → {out_bar}")
