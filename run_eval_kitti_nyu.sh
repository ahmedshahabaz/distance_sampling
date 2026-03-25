#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate test
python3 eval_kitti.py DA True eigen
python3 eval_kitti.py DA True validation
python3 eval_nyu.py DA True
conda deactivate

conda activate midas

python3 eval_kitti.py MD True eigen
python3 eval_nyu.py MD True

conda deactivate

# METRIC DEPTH MODEL EVALUATION

conda activate test
python3 eval_kitti.py DA False eigen
python3 eval_kitti.py DA False validation
python3 eval_nyu.py DA False
conda deactivate

conda activate midas
python3 eval_kitti.py AB False eigen
python3 eval_nyu.py AB False

python3 eval_kitti.py ZD False eigen
python3 eval_kitti.py ZD False validation
python3 eval_nyu.py ZD False
conda deactivate


conda activate test
python3 eval_kitti.py ZD False eigen
python3 eval_kitti.py ZD False validation
python3 eval_nyu.py ZD False

python3 eval_kitti.py UD False eigen
python3 eval_nyu.py UD False

python3 eval_kitti.py UN False eigen
python3 eval_nyu.py UN False
conda deactivate








