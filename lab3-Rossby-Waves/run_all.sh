#!/bin/bash

for nl in namelists/*; do
    ./st_venant.x "$nl"
done

echo " "
echo "=============== COMPELETED ./st_venant.x FOR THE 'NAMELISTS/' FOLDER ============"
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


