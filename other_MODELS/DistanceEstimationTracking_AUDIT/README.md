# DistanceEstimationTracking_AUDIT

This folder is used to run the AUDIT baseline on the animal-distance images used in this project.

The main entry point is `eval_animal_distances_AUDIT.py`. It is used to:

1. take one animal image at a time from `./data/`
2. obtain or load a depth prediction
3. align that depth with the AUDIT PVCNN model
4. detect the animal with MegaDetector
5. segment the animal with SAM
6. estimate the animal distance from the aligned depth map
7. compare the estimate against the ground-truth distances from `Animal_Distances.xlsx`
8. save an Excel file for comparison with the main method in this repository

## What This Is For

Use this folder when you want AUDIT-based comparison results for the animal-distance experiment.

The output Excel files are intended to be compared with the main project output from `eval_animal_distances.py`. In particular, `animal_dist_plots.py` expects AUDIT result files named like:

- `animal_dist_rslt_AUDIT_DA.xlsx`
- `animal_dist_rslt_AUDIT_DPT.xlsx`

## Important Inputs

Run the script from inside this folder:

```bash
cd other_MODELS/DistanceEstimationTracking_AUDIT
```

The script expects these inputs:

- `./data/`
  Place the animal images here.
- `./inference_test/Animal_Distances.xlsx`
  This is the ground-truth table used to build the final AUDIT result sheet.
- `./align_weights.pth`
  AUDIT alignment weights.
- `./DPT/weights/dpt_large-midas-2f21e586.pt`
  Needed when running in `DPT` mode.

## Input Image Naming

The script matches predictions to the Excel file by image name and site code.

Use names of the form:

```text
DSCF0009_40KL.JPG
```

Important:

- keep the site code in the filename
- use the `.JPG` extension in uppercase, because the script currently looks for `*.JPG`

## Running The Script

### Option 1: AUDIT with DPT

```bash
python eval_animal_distances_AUDIT.py --model DPT
```

In this mode, the script can generate raw DPT depth files itself if they are not already present in:

```text
./temp/DPT_raw/
```

It then aligns them and writes aligned outputs to:

```text
./inference_test/algn_out_DPT/
```

### Option 2: AUDIT with Depth Anything

```bash
python eval_animal_distances_AUDIT.py --model DA
```

In this mode, the script expects raw Depth Anything predictions to already exist in:

```text
./temp/DA_raw/
```

These files are `.npy` depth arrays, one per image. The script then aligns them and writes aligned outputs to:

```text
./inference_test/algn_out_DA/
```

## Outputs

After running, the main outputs are:

- `./inference_test/animal_dist_rslt_AUDIT_DPT.xlsx`
- `./inference_test/animal_dist_rslt_AUDIT_DA.xlsx`

These spreadsheets contain:

- `Site`
- `file_name`
- `Mask Dist`
- `GT Dist`
- `diff_err`
- `abs_err`
- `abs_rel`

The script also creates intermediate files in:

- `./temp/`
- `./inference_test/algn_out_DPT/`
- `./inference_test/algn_out_DA/`

## Practical Notes

- This script is for still-image animal-distance evaluation, not the reference-point video evaluation.
- If an image is present in `./data/` but not represented correctly in `Animal_Distances.xlsx`, it will not be matched properly in the final report.
- If you want to compare AUDIT with the main method in this repository, copy the final Excel outputs to the location expected by `animal_dist_plots.py`.
