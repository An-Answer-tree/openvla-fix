source /home/szliutong/.zshrc
cd /home/szliutong/Projects/OpenVLA_LIBERO_FIX/openvla
conda activate openvla-fix

python experiments/robot/libero/run_libero_eval_fixed.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
    --task_suite_name libero_10 \
    --local_log_dir ./experiments/logs/fixed/10 \
    --rollout_dir ./rollouts/fixed/10
