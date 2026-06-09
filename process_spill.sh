
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "DEBUG: Running .sh inside $SCRIPT_DIR"

cd ${SCRIPT_DIR}

source define_envs.sh


cd ${WORKING_DIR}

timeout 30s ./fullexecution.sh "$@" 2>&1 | tee ${LOGS_DIR}/log_run${1}_spill${2}_typeis${3}.log
