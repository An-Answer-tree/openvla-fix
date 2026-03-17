source /home/szliutong/.zshrc
cd /home/szliutong/Projects/OpenVLA_LIBERO_FIX/openvla
conda activate openvla-fix

python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
    --task_suite_name libero_object \
    --local_log_dir ./experiments/logs/origin/object \
    --rollout_dir ./rollouts/origin/object
