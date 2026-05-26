#!/bin/bash

echo " "
echo "=============== WE ARE RUNNING ./st_venant.x FOR THE 'NAMELISTS/' FOLDER ============"
echo " "

for nl in namelists/*; do
    # extract output folder from namelist
    outdir=$(grep "data_name" "$nl" | sed "s/.*= *'//" | sed "s/'.*//")

    echo "  Will run: $nl"
    echo "  Results will be in: $outdir"
    echo " "
done

echo "================================================================================="
echo " "    

for nl in namelists/*; do
    ./st_venant.x "$nl"
done

echo " "
echo "=============== COMPLETED ./st_venant.x FOR THE 'NAMELISTS/' FOLDER ============"
echo " "

for nl in namelists/*; do
    # extract output folder from namelist
    outdir=$(grep "data_name" "$nl" | sed "s/.*= *'//" | sed "s/'.*//")

    echo "  Completed for: $nl"
    echo "  Results is in: $outdir"
    echo " "
done

echo "================================================================================="
echo " "    

echo "=============== RUNNING STREAMFUNCTION PLOTS ==============="
echo " "

# initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# activate environment
conda activate vattensnok

# run plotting script
python stream_funtion.py

echo " "
echo "=============== STREAMFUNCTION PLOTS COMPLETED ==============="
echo " "
