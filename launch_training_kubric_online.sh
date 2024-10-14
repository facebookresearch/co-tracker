#!/bin/bash

EXP_DIR=$1
EXP_NAME=$2
DATE=$3
DATASET_ROOT=$4
NUM_STEPS=$5


echo `which python`

mkdir -p ${EXP_DIR}/${DATE}_${EXP_NAME}/logs/;
mkdir ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3;
find . \( -name "*.sh" -o -name "*.py" \) -type f -exec cp --parents {} ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3 \;

export PYTHONPATH=`(cd ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3 && pwd)`:`pwd`:$PYTHONPATH
sbatch --comment=${EXP_NAME} --partition=learn --account=repligen --qos=repligen --time=39:00:00 --gpus-per-node=8 --nodes=4 --ntasks-per-node=8 \
--job-name=${EXP_NAME} --cpus-per-task=10 --signal=USR1@60 --open-mode=append \
--output=${EXP_DIR}/${DATE}_${EXP_NAME}/logs/%j_%x_%A_%a_%N.out \
--error=${EXP_DIR}/${DATE}_${EXP_NAME}/logs/%j_%x_%A_%a_%N.err \
--wrap="srun --label python ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3/train_on_kubric.py --batch_size 1 \
--num_steps ${NUM_STEPS} --ckpt_path ${EXP_DIR}/${DATE}_${EXP_NAME} --model_name cotracker_three \
--save_freq 200 --sequence_len 64 --eval_datasets tapvid_davis_first tapvid_stacking \
--traj_per_sample 384 --sliding_window_len 16 --train_datasets kubric \
--save_every_n_epoch 5 --evaluate_every_n_epoch 5 --model_stride 4 --dataset_root ${DATASET_ROOT} --num_nodes 4 \
--num_virtual_tracks 64 --mixed_precision \
--corr_radius 3 --wdecay 0.0005 --linear_layer_for_vis_conf --validate_at_start --add_huber_loss"
