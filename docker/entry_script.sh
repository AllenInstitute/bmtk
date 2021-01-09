#!/bin/bash
set -e

if [[ $1 = "nrnivmodl" ]]; then
    shift
    cd /home/shared/workspace/mechanisms
    nrnivmodl $@
elif [[ $1 = "python" ]]; then
    shift
    cd /home/shared/workspace
    python $@
elif [[ $1 = "jupyter" ]]; then
    shift
    jupyter notebook --allow-root --ip=* --port 8888 --no-browser --notebook-dir /home/shared --NotebookApp.token=""
else
    cd /home/shared/workspace
    python $@
fi
