#!/bin/bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=2

# 输入和输出目录设置
INPUT_DIR="/data/lcx/RRDataset_original_train_val"
OUTPUT_DIR="/data/lcx/DNF_ORIGINAL_TRAIN_VAL"

# 设置工作目录
cd ./DNF-main

# 处理 train/ai 文件夹
echo "Processing train/ai folder..."
mkdir -p "${OUTPUT_DIR}/train/ai"
python compute_dnf.py \
    --dataroot "${INPUT_DIR}/train/ai" \
    --save_root "${OUTPUT_DIR}/train/ai" \
    --batch_size 16 \
    --num_threads 1 \
    --gpu_ids "0" \
    --config "config.yaml" \
    --diffusion_ckpt ./weights/diffusion/model-2388000.ckpt

# 处理 train/real 文件夹
echo "Processing train/real folder..."
mkdir -p "${OUTPUT_DIR}/train/real"
python compute_dnf.py \
    --dataroot "${INPUT_DIR}/train/real" \
    --save_root "${OUTPUT_DIR}/train/real" \
    --batch_size 16 \
    --num_threads 1 \
    --gpu_ids "0" \
    --config "config.yaml" \
    --diffusion_ckpt ./weights/diffusion/model-2388000.ckpt

# 处理 val/ai 文件夹
echo "Processing val/ai folder..."
mkdir -p "${OUTPUT_DIR}/val/ai"
python compute_dnf.py \
    --dataroot "${INPUT_DIR}/val/ai" \
    --save_root "${OUTPUT_DIR}/val/ai" \
    --batch_size 16 \
    --num_threads 1 \
    --gpu_ids "0" \
    --config "config.yaml" \
    --diffusion_ckpt ./weights/diffusion/model-2388000.ckpt

# 处理 val/real 文件夹
echo "Processing val/real folder..."
mkdir -p "${OUTPUT_DIR}/val/real"
python compute_dnf.py \
    --dataroot "${INPUT_DIR}/val/real" \
    --save_root "${OUTPUT_DIR}/val/real" \
    --batch_size 16 \
    --num_threads 1 \
    --gpu_ids "0" \
    --config "config.yaml" \
    --diffusion_ckpt ./weights/diffusion/model-2388000.ckpt

echo "Processing complete!"