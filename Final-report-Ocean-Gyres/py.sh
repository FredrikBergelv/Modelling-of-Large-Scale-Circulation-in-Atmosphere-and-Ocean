#!/bin/bash


echo "=============== RUNNING STREAMFUNCTION PLOTS ==============="
echo " "

# initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# activate environment
conda activate vattensnok

# run plotting script
for file in *.py; do
    python "$file"
done


echo " "
echo "=============== STREAMFUNCTION PLOTS COMPLETED ==============="
echo " "
