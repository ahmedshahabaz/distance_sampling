import argparse

import numpy as np


MODEL_NAME_DICT = {
    "MD": "MIDAS",
    "DA": "DepthAnything",
    "ZD": "ZoeDepth",
    "UD": "UniDepth",
    "UN": "UniK3D",
    "AB": "AdaBins",
}

VALID_MODELS = list(MODEL_NAME_DICT.keys())

# Models that only support one depth type.
# DA supports both; all others are restricted to one.
METRIC_ONLY_MODELS   = {"ZD", "UD", "UN", "AB"}   # metric depth only
RELATIVE_ONLY_MODELS = {"MD"}                       # relative depth only

# ── Default dataset paths ──────────────────────────────────────────────────────

_DS_ROOT = "."

_EVAL_ROOT_DIR_       = f"{_DS_ROOT}/REPORTS/ref_pts_eval"
DEFAULT_ROOT_DIR       = f"{_DS_ROOT}/outputs/depth_preds"
DEFAULT_DATA_DIR       = f"{_DS_ROOT}/DATA/VIDEOS"
DEFAULT_MASK_DIR       = f"{_DS_ROOT}/outputs/animal_seg_mask"
DEFAULT_ANIMAL_BOX_DIR = f"{_DS_ROOT}/outputs/animal_bboxes"


def validate_model_depth_type(parser, model_name, use_relative):
    """
    Check that the model/depth-type combination is valid.
    Calls parser.error() (exits with code 2) if not.

    Valid combinations
    ------------------
    DA  : --relative  or  --metric
    MD  : --relative  only
    ZD, UD, UN, AB : --metric  only
    """
    if model_name in METRIC_ONLY_MODELS and use_relative:
        parser.error(
            f"'{model_name}' is a metric-depth-only model. "
            f"Use --metric instead of --relative."
        )
    if model_name in RELATIVE_ONLY_MODELS and not use_relative:
        parser.error(
            f"'{model_name}' is a relative-depth-only model. "
            f"Use --relative instead of --metric."
        )


def add_depth_args(parser):
    """
    Add the standard depth-model arguments to an existing ArgumentParser.

    Positional
    ----------
    model        – model shortcode: DA | MD | ZD | UD | UN | AB
                   MD       : --relative only
                   ZD/UD/UN/AB : --metric only
                   DA       : both supported

    Required (mutually exclusive)
    -----------------------------
    --relative / --rltv  – use relative depth (calibration always applied)
    --metric             – use metric depth

    Optional
    --------
    --calb           – apply disparity calibration (metric only; always on for relative depth)
    --root-dir DIR   – root directory for depth predictions (default: DEFAULT_ROOT_DIR)
    --data-dir DIR   – directory containing video files and annotations (default: DEFAULT_DATA_DIR)
    """
    parser.add_argument(
        "model",
        choices=VALID_MODELS,
        help="Model shortcode.",
    )
    depth_type = parser.add_mutually_exclusive_group(required=True)
    depth_type.add_argument(
        "--relative", "--rltv",
        action="store_true",
        dest="relative",
        help="Use relative depth. Calibration is always applied.",
    )
    depth_type.add_argument(
        "--metric",
        action="store_true",
        dest="metric",
        help="Use metric depth.",
    )
    parser.add_argument(
        "--calb",
        action="store_true",
        help="Apply disparity calibration (metric depth only; always on for relative depth).",
    )
    parser.add_argument(
        "--root-dir",
        default=DEFAULT_ROOT_DIR,
        metavar="DIR",
        help="Root directory for depth predictions. (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        metavar="DIR",
        help="Directory containing video files and annotations. (default: %(default)s)",
    )
    return parser


def add_dataset_path_args(parser):
    """
    Add per-script dataset path arguments (mask dir, animal bbox dir).

    Call this after add_depth_args() for scripts that load segmentation masks
    or animal bounding boxes (eval_ref_point_distances, plot_ref_pts_depth_preds).

    Optional
    --------
    --mask-dir DIR        – directory of per-frame animal segmentation masks
    --animal-box-dir DIR  – directory of animal bounding-box JSON files
    """
    parser.add_argument(
        "--mask-dir",
        default=DEFAULT_MASK_DIR,
        metavar="DIR",
        help="Directory of per-frame animal segmentation masks. (default: %(default)s)",
    )
    parser.add_argument(
        "--animal-box-dir",
        default=DEFAULT_ANIMAL_BOX_DIR,
        metavar="DIR",
        help="Directory of animal bounding-box JSON files. (default: %(default)s)",
    )
    return parser


def parse_depth_cli(description=""):
    """
    Build and parse the standard depth-model CLI.

    Convenience wrapper around add_depth_args() for scripts that only need
    the standard arguments (model, --relative/--metric, --calb, --root-dir,
    --data-dir).  Scripts that need additional flags (e.g. --save-videos)
    should call add_depth_args() on their own ArgumentParser instead.

    Returns
    -------
    model_name   : str
    use_relative : bool
    use_calb     : bool  – always True when use_relative is True
    root_dir     : str   – path to depth-predictions root (--root-dir)
    data_dir     : str   – path to video/annotation directory (--data-dir)
    """
    parser = argparse.ArgumentParser(description=description)
    add_depth_args(parser)
    args = parser.parse_args()
    use_relative = args.relative
    use_calb = True if use_relative else args.calb
    validate_model_depth_type(parser, args.model, use_relative)
    return args.model, use_relative, use_calb, _EVAL_ROOT_DIR_, args.data_dir


def print_run_banner(model_name, use_relative, use_calb, output_dir):
    print()
    print("------------------")
    print("Model   :", MODEL_NAME_DICT[model_name])
    print("Type    :", "Relative Depth" if use_relative else "Metric Depth")
    if use_calb:
        print("Calibrating preds!")
    print(output_dir)
    print("------------------")
    print()


def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    return obj


def get_ref_point_circles(video_data):
    """Return reference-point entries from a video record."""
    # Backward compatibility: older JSON outputs stored circles under "lines".
    return video_data.get("circles", video_data.get("lines", []))
