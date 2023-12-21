python train.py \
--batch_size 1 \
--num_workers 16 \
--num_steps 4 \
--ckpt_path ./checkpoints \
--dataset_root /hhd3/GroupProject2023Fall/datasets/point_odyssey_demo \
--model_name cotracker \
--save_freq 1 \
--sequence_len 16 \
--traj_per_sample 16 \
--sliding_window_len 8 \
--updateformer_space_depth 6 \
--updateformer_time_depth 6 \
--save_every_n_epoch 1 \
--evaluate_every_n_epoch 1000 \
--model_stride 4 \
--gpus 0 1 2 3
# --eval_datasets tapvid_davis_first badja \