# Monocular Depth Estimation for Wildlife Distance Sampling

Code for the paper:

> [**From Relative to Metric: Calibrating AI-based Monocular Depth Learning Models for Distance Sampling in Wildlife Monitoring Applications**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5537631)

This repository evaluates monocular depth estimation models for estimating animal distances in camera-trap videos, using a distance-sampling field protocol as ground truth. The pipeline covers depth prediction, calibration, reference-point evaluation, animal distance estimation, and statistical reporting.

Unless noted otherwise, run the commands below from the project root (`distance_sampling/`).

---

## Supported Models

| Code | Model | Type |
|------|-------|------|
| `DA` | Depth Anything V2 | Relative |
| `MD` | MiDaS | Relative |
| `ZD` | ZoeDepth | Metric |
| `UD` | UniDepth | Metric |
| `UN` | UniK3D | Metric |
| `AB` | AdaBins | Metric |

---

## Repository Structure

```
distance_sampling/
│
├── DATA/
│   ├── VIDEOS/               # Site subdirectories with .AVI video files
│   │   ├── 31KL/
│   │   ├── 33LD/
│   │   └── ...
│   ├── animal_dist_imgs/     # Still images for animal distance estimation
│   ├── Animal_Distances.xlsx # Ground-truth animal distances
│   └── ideal_calibration_results.json
│
├── outputs/
│   └── depth_preds/
│       ├── rltv_depths/      # Relative depth predictions (.npy per video)
│       └── metric_depths/    # Metric depth predictions (.npy per video)
│
├── checkpoints/              # Model weight files
├── depth_anything_v2/        # Depth Anything V2 model architecture
├── other_MODELS/             # Third-party model code (MiDaS, ZoeDepth, etc.)
│   ├── AdaBins/
│   ├── DA_metric/
│   ├── DistanceEstimationTracking_AUDIT/
│   ├── MiDaS/
│   ├── UniDepth/
│   ├── UniK3D/
│   └── ZoeDepth/
└── utils/                    # Shared utilities
```

---

## Pipeline Overview

### Step 1 — Save Depth Predictions

Run inference for each model and save per-frame depth maps as `.npy` files (shape: `num_frames × H × W`).

**Depth Anything V2 (relative)** — run from project root:
```bash
python save_preds_DA.py [--outdir DIR] [--data-dir DIR]
```

**Other models** — each has its own script under `other_MODELS/<MODEL>/`:
```bash
# e.g.
cd other_MODELS/ZoeDepth && python save_preds_ZoeDepth.py [--outdir DIR] [--data-dir DIR]
```

Or run all models sequentially (activates the correct conda environment per model):
```bash
bash run_save_preds.sh
```

> The script uses two conda environments: `midas` (for ZoeDepth, MiDaS, AdaBins) and `test` (for DA, UniDepth, UniK3D, DA_metric).

---

### Optional — Pre-compute Animal Bounding Boxes and Segmentation Masks

These steps are **not required** before running the evaluation scripts. If bounding boxes or masks are not found on disk, they are computed on-the-fly during evaluation.

Pre-computing and caching them is recommended when running experiments repeatedly, as it avoids re-running detection and segmentation for every evaluation pass.

**Detect animals** using MegaDetector V6 and save per-frame bounding boxes as JSON files:
```bash
python generate_animal_bboxes.py
```

**Generate segmentation masks** using SAM from the bounding boxes, saved per-frame and used during calibration to exclude occluded reference patches:
```bash
python generate_animal_masks.py
```

`generate_animal_bboxes.py` also writes annotated preview videos to `./outputs/detected_vids/`.

---

### Step 2 — Evaluate Reference-Point Distances

For each video, fits a per-frame disparity calibration (scale + shift) against ground-truth distances at known reference-point circles, then evaluates depth accuracy.

```bash
python eval_ref_point_distances.py <model> --relative | --metric [--calb]
                                   [--root-dir DIR] [--data-dir DIR]
                                   [--mask-dir DIR] [--animal-box-dir DIR]
                                   [--bg-fit] [--bg-fit-method {least_squares,ransac}]
```

- `--relative` — for relative depth models (calibration always applied)
- `--metric` — for metric depth models
- `--calb` — additionally apply disparity calibration for metric models
- `--bg-fit` — align per-frame background disparity to site mean (DA model)

Output JSON written to the model's prediction directory:
- `ref_point_stats.json` — uncalibrated
- `ref_point_stats_calb.json` — calibrated

Run all model/mode combinations:
```bash
bash run_eval_ref_point.sh
```

---

### Step 3 — Evaluate Animal Distances

Estimates metric distances to detected animals in still images using depth predictions, calibrated against visible reference-point circles.

```bash
python eval_animal_distances.py [--excel-file FILE] [--output-root DIR]
                                [--report-path FILE]
                                [--ideal-calibration-json FILE]
                                [--min-visible-patches N]
```

- Input images/videos are expected in `./DATA/animal_dist_imgs/` named `{FileNumber}_{Site}.EXT`
- Ground-truth distances and site metadata come from `Animal_Distances.xlsx`
- Falls back to pre-computed ideal calibration when fewer than `--min-visible-patches` reference patches are visible
- This script currently uses the Depth Anything V2 pipeline defined in `infer_utils.py`; `--model-name` is kept only as an optional naming field

---

### Step 4 — Reporting and Visualisation

**Overall error summary (filtered / by distance bin):**
```bash
python report_ref_pts_stats.py <model> --relative | --metric [--calb]
python report_ref_pts_stats_dist_bins.py <model> --relative | --metric [--calb]
```

**Time-of-day (TOD) box plot:**
```bash
python report_ref_pts_TOD_box_plot.py <model> --relative | --metric [--calb]
```

This script also prints per-time-of-day summary statistics to the terminal. A separate TOD hypothesis-test script is not included in this repository snapshot.

**Annotated video output and variance/VMR plots:**
```bash
python plot_ref_pts_depth_preds.py <model> --relative | --metric [--calb] [--save-videos]
```
Note: this plotting script currently runs on a hard-coded subset of sites listed near the top of the file. Edit the `SITES` list there if you want to process more or fewer sites.

**Animal distance result plots (for paper figures):**
```bash
python animal_dist_plots.py
```

**Depth vs. Z geometry illustration:**
```bash
python depth_vs_Z.py
```

---

### Benchmark Evaluation (KITTI / NYU)

`eval_kitti.py` and `eval_nyu.py` evaluate models on standard depth benchmarks.

```bash
python eval_kitti.py <model> <use_relative> [data_split]   # data_split: eigen | val | test
python eval_nyu.py  <model> <use_relative>
```

Run all model/split combinations:
```bash
bash run_eval_kitti_nyu.sh
```

#### Data Setup

> **Note:** KITTI and NYU data are not provided in this repository. To run these evaluations, place the datasets as follows:
>
> - **KITTI Eigen split** — raw RGB images and ground-truth depth maps following the standard Eigen et al. split. The script expects a directory structure of `<kitti_root>/<date>/<sequence>/`.
> - **NYU Depth V2** — the standard test set (654 images). The script expects `.npy` or `.mat` files in a flat directory.
>
> By default, the scripts look under `./DATA/kitti_data/` and `./DATA/nyu_data/`. If you keep the datasets elsewhere, update the path constants near the top of `eval_kitti.py` and `eval_nyu.py`.

---

## Utility Modules

| File | Description |
|------|-------------|
| `infer_utils.py` | Model loading and depth inference for Depth Anything V2 and HuggingFace models |
| `utils/calb_utils.py` | Calibration, error metrics, SAM masking, stats aggregation, frame annotation |
| `utils/pipeline_utils.py` | Directory resolution, video/depth file pairing, frame loading, depth normalisation |
| `utils/analysis_utils.py` | Pixel-wise variance/VMR computation, temporal depth plots |
| `utils/script_utils.py` | Shared CLI parsing and run-banner helpers |

---

## Conda Environments

The codebase uses two conda environments. Full exported environment files are provided for reproducibility:

```bash
conda env create -f environment_test.yml   # test  — DA, UniDepth, UniK3D, DA_metric
conda env create -f environment_midas.yml  # midas — MiDaS, ZoeDepth, AdaBins
```

| Environment | Models |
|-------------|--------|
| `test` (Python 3.10, PyTorch 2.2, CUDA 12.1) | Depth Anything V2, UniDepth, UniK3D, DA_metric |
| `midas` (Python 3.10, PyTorch 1.13, CUDA 11.7) | MiDaS, ZoeDepth, AdaBins |

---

## Citation

If you use this code, please cite:

```
@article{shahabaz5537631relative,
  title={From Relative to Metric: Calibrating AI-based Monocular Depth Learning Models for Distance Sampling in Wildlife Monitoring Applications},
  author={Shahabaz, Ahmed and Toczyd{\l}owska, Joanna and Gula, Roman and Theuerkauf, J{\"o}rn and Sarkar, Sudeep},
  journal={Available at SSRN 5537631}
}
```
*(BibTeX to be updated with the accepted paper details.)*

---

## Data Availability

Due to ongoing ecological research projects and conservation considerations, the full camera-trap video dataset cannot be made publicly available. A small subset of example videos used to demonstrate the pipeline is provided in this repository. Additional data may be made available from the authors upon request.

---

## Acknowledgements

This research was supported in part by the US National Science Foundation grant IIS 1956050. The field research was funded by SAVE – Wildlife Conservation Fund, Poland.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
