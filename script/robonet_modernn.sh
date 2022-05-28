export CUDA_VISIBLE_DEVICES=3
cd ..
nohup python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name robonet \
    --train_data_paths1 /data/RoboNet \
    --valid_data_paths1 /data/RoboNet \
    --save_dir checkpoints/robonet_modernn \
    --gen_frm_dir results/robonet_modernn \
    --model_name modernn \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 3 \
    --input_length 5 \
    --total_length 15 \
    --num_hidden 64,64,64,64 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 1 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr 0.0001 \
    --batch_size 8 \
    --max_iterations 120000 \
    --display_interval 100 \
    --test_interval 1000 \
    --snapshot_interval 1000 > logs/robonet_modernn.log 2>&1 &


