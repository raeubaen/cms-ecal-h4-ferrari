#!/bin/bash

set -e

# 1. Check for the run number argument
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <run_number> <energy>"
    exit 1
fi
RUN=$1
ENERGY=$2

# Define paths
MAP_CSV="maps/tb_map.csv"
GAIN_CSV="plotter/resgainratios.csv"
RERECO_SCRIPT="ferrari/core/offline-scripts/re-reco.sh"
PLOTTER_SCRIPT="plotter/plotter_resolution_singlecrystal.py"
RECO_DIR=""
FIT_OUTPUT_DIR="/eos/user/l/lfaiella/www/h4dqm/ECAL_TB_2026_latestDQM/GainRatio/FitPlots"

# Initialize/overwrite the results CSV with a clean header
echo "gain_ratio,resolution" > "$GAIN_CSV"

# 2. Setup loop factors using bc for floating-point math
START=0.8
STEP=0.04
END=1.2

CURRENT=$START
while [ "$(echo "$CURRENT <= $END + 0.0001" | bc)" -eq 1 ]; do

    # Calculate exact gain ratio for this loop
    GAIN_RATIO=$(echo "scale=4; 10.2 * $CURRENT" | bc)
    echo "--------------------------------------------------"
    echo "Factor: $CURRENT | Gain Ratio: $GAIN_RATIO"
    echo "--------------------------------------------------"

    # 3. Modify row 241 (index 240) of high_over_low_gain_ratio in-place
    python3 -c "
import pandas as pd
df = pd.read_csv('$MAP_CSV')
df.loc[240, 'high_over_low_gain_ratio'] = $GAIN_RATIO
df.to_csv('$MAP_CSV', index=False)
"

    # 4. Run the re-reco script
    echo "Running re-reco for run $RUN..."
    bash "$RERECO_SCRIPT" "$RUN"

    # 5. Run the Python plotter and capture its output
    echo "Running resolution plotter..."
    
    # CRITICAL STEP: Captures whatever python prints out
    RESOLUTION_OUTPUT=$(python3 "$PLOTTER_SCRIPT" -r $RUN -e $ENERGY -g $GAIN_RATIO -f $FIT_OUTPUT_DIR)

    # If your python script prints extra text logs alongside the number, 
    # we can isolate just the floating-point number using grep/awk:
    RESOLUTION=$(echo "$RESOLUTION_OUTPUT" | grep -oE '[0-9]+\.[0-9]+' | tail -n 1)

    # Fallback check: If grep didn't find a decimal, just use the raw output stripped of whitespace
    if [ -z "$RESOLUTION" ]; then
        RESOLUTION=$(echo "$RESOLUTION_OUTPUT" | xargs)
    fi

    echo "Extracted Resolution: $RESOLUTION"

    # 6. Save both values cleanly as a new row in the CSV
    echo "${GAIN_RATIO},${RESOLUTION}" >> "$GAIN_CSV"

    # Advance the loop factor
    CURRENT=$(echo "scale=2; $CURRENT + $STEP" | bc)
done

echo "--------------------------------------------------"
echo "All iterations finished. Data saved to $GAIN_CSV"
