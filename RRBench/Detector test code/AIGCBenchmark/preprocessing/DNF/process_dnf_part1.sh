#!/bin/bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=2


INPUT_DIR="/data/RRDataset_original_train_val"
OUTPUT_DIR="/data/DNF_ORIGINAL_TRAIN_VAL"


cd ./DNF-main


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