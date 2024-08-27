#!/bin/bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${1:-0}
MASTER_PORT=${2:-4885}

python train_piaa.py \
        --dataset_name "REALCURDataset" \
        --root_dir "./REAL-CUR" \
        --label_file 'test_list.csv' \
        --output_dir "./logs" \
        --num_models 6 \
        --exp "worker_${w}" \
        --worker_idx $w \
        --softmax_temp 10.0 \
        --k 100 \
        --seed 0 \
        --clip_model 'ViT-L-14' \
        --clip_pretrained 'datacomp_xl_s13b_b90k' \
        --batch_size 32 \
        --image_size 224 \
        --accumulation_steps 1 \
        --save_every 100 \
        --num_steps 500 \
        --weight_decay 0.0 \
        --lr 1.0e-2 \
        --min_lr 1.0e-3 \
        --warmup_steps 0 \
        --model_type "3fc" \
        --precision "full" \
        --save_csv \
        --ignore_warnings
