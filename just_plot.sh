#!/bin/bash

# --- launch settings with beam|laser as input parameter ---
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <run_number> <spill_number> <spill_type>"
    return
fi
RUN=$1
SPILL=$(printf "%04d" $((10#$2)))
OPTION=$3

echo run: $RUN
echo spill: $SPILL
echo SPILL_TYPE: $OPTION

SPILL_STR="${SPILL}_${OPTION}"

mkdir ${PLOT_MAIN_FOLDER}/run_$RUN/

RECO_FILE=${RECO_UNPACKED_OUTDIR}/reco_dqm/run_$RUN/${RUN}_${SPILL}_reco.root
PLOT_SPILL_FOLDER=${PLOT_MAIN_FOLDER}/run_$RUN/spill_${SPILL_STR}

python3 -m ferrari_core.offline-scripts.plot_uproot_numpysyntax -i ${RECO_FILE} -po ${PLOT_SPILL_FOLDER} -j ${JSON_CONF} -opt $OPTION

