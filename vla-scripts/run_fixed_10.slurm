#!/bin/bash
#SBATCH --job-name=openvla-libero-fixed-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=high_priority
#SBATCH --output=/home/liutong/Projects/OpenVLA_LIBERO_FIX/openvla/logs/fixed_10/%x-%j.log

set -eo pipefail

source /home/liutong/.bashrc
conda activate openvla-fix

export http_proxy="http://localhost:7878"
export https_proxy="http://localhost:7878"

mkdir -p /opt/liutong/openvla-runs-fixed/10
mkdir -p /opt/liutong/openvla-adapter-tmp-fixed/10

cd /home/liutong/Projects/OpenVLA_LIBERO_FIX/openvla

echo "[$(date '+%F %T')] job=${SLURM_JOB_ID:-N/A} node=$(hostname) gpus=1 fixed training setting on 10"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=1 \
  vla-scripts/finetune_fix.py \
  --dataset_name libero_10_no_noops \
  --wandb_project openvla-origin-vs-fixed-10 \
  --run_root_dir /opt/liutong/openvla-runs-fixed/10 \
  --adapter_tmp_dir /opt/liutong/openvla-adapter-tmp-fixed/10
