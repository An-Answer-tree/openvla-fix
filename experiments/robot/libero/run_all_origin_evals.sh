#!/usr/bin/env zsh

set -euo pipefail

cd /home/szliutong/Projects/OpenVLA_LIBERO_FIX/openvla

for script in \
    experiments/robot/libero/eval_origin_spatial.sh \
    experiments/robot/libero/eval_origin_object.sh \
    experiments/robot/libero/eval_origin_goal.sh \
    experiments/robot/libero/eval_origin_10.sh
do
    echo "Running ${script}..."
    zsh "${script}"
done
