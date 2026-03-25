#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh


conda activate test
echo "*** DEPTH ANYTHING ***"
python3 eval_ref_point_distances.py DA --relative
echo "------------------"
python3 eval_ref_point_distances.py DA --metric
echo "------------------"
python3 eval_ref_point_distances.py DA --metric --calb
echo "------------------"
conda deactivate
echo "------------------"

conda activate midas

echo "*** MIDAS ***"
python3 eval_ref_point_distances.py MD --relative
echo "------------------"

echo "*** ADABINS ***"
python3 eval_ref_point_distances.py AB --metric
echo "------------------"
python3 eval_ref_point_distances.py AB --metric --calb
echo "------------------"

conda deactivate


conda activate test
echo "*** UNIK3D ***"
python3 eval_ref_point_distances.py UN --metric
echo "------------------"
python3 eval_ref_point_distances.py UN --metric --calb
echo "------------------"

echo "*** UNIDEPTH ***"
python3 eval_ref_point_distances.py UD --metric
echo "------------------"
python3 eval_ref_point_distances.py UD --metric --calb
echo "------------------"

echo "*** ZOEDEPTH ***"
python3 eval_ref_point_distances.py ZD --metric
echo "------------------"
python3 eval_ref_point_distances.py ZD --metric --calb
echo "------------------"

conda deactivate
