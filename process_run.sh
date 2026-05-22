cd ${WORKING_DIR}

timeout 120s ./fullexecution.sh "$@" 2>&1 | tee ${LOGS_DIR}/log_run${1}_spill${2}_typeis${3}.log
