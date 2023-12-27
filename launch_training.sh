#!/bin/bash

EXP_DIR=$1
EXP_NAME=$2
DATE=$3
DATASET_ROOT=$4
NUM_STEPS=$5


echo `which python`

mkdir -p ${EXP_DIR}/${DATE}_${EXP_NAME}/logs/;

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
sbatch --comment=${EXP_NAME} --partition=learn --time=39:00:00 --gpus-per-node=8 --nodes=4 --ntasks-per-node=8 \
--job-name=${EXP_NAME} --cpus-per-task=10 --signal=USR1@60 --open-mode=append \
--output=${EXP_DIR}/${DATE}_${EXP_NAME}/logs/%j_%x_%A_%a_%N.out \
--error=${EXP_DIR}/${DATE}_${EXP_NAME}/logs/%j_%x_%A_%a_%N.err \
--wrap="srun --label python ./train.py --batch_size 1 \
--num_steps ${NUM_STEPS} --ckpt_path ${EXP_DIR}/${DATE}_${EXP_NAME} --model_name cotracker \
--save_freq 200 --sequence_len 24 --eval_datasets dynamic_replica tapvid_davis_first \
--traj_per_sample 768 --sliding_window_len 8 \
--save_every_n_epoch 10 --evaluate_every_n_epoch 10 --model_stride 4 --dataset_root ${DATASET_ROOT} --num_nodes 4 \
--num_virtual_tracks 64"
