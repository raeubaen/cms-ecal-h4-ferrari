RUN=$1
NSPILLS=$2
option=$3

LOGS_FOLDER="${PLOT_MAIN_FOLDER}/re-plot-logs/"

echo nspills: $NSPILLS

mkdir $LOGS_FOLDER

echo "LOGS in " ${LOGS_FOLDER}

mkdir $LOGS_FOLDER/log_${RUN}

for spill in $(seq 1 $NSPILLS); do
    echo $spill
    cd $WORKING_DIR

    # Launch background job for this actual spill
    bash -c "source just_plot.sh $RUN $spill $option >  $LOGS_FOLDER/log_${RUN}/log_${RUN}_${spill}.log 2>&1 &"

    while true; do
        running=$(ps aux | grep "just_plot.sh" | grep -v grep | wc -l)
        if (( running < 12 )); then
            break
        fi
        sleep 1
    done

done


ls -1d /eos/user/m/mcampana/www/h4dqm/ECAL_TB_2026/run_${RUN}/spill_0* > /tmp/hadd_${RUN}

echo for hadding, suggest to do: "export HADD_NOW_DIRS=/tmp/hadd_${RUN}"

