#!/usr/bin/env zsh

set -euo pipefail

cd /home/szliutong/Projects/OpenVLA_LIBERO_FIX/openvla

for script in \
    experiments/robot/libero/eval_fixed_spatial.sh \
    experiments/robot/libero/eval_fixed_object.sh \
    experiments/robot/libero/eval_fixed_goal.sh \
    experiments/robot/libero/eval_fixed_10.sh
do
    echo "Running ${script}..."
    zsh "${script}"
done
