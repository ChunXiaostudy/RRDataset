#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# Model weights path
WEIGHTS="./weights/preprocessing/sidd_rgb.pth"

# Base directories
INPUT_DIR="/data/lcx/RRDataset_original_train_val"
OUTPUT_DIR="/data/lcx/LNP_ORIGINAL_TRAIN_VAL"

# Process train/ai folder
echo "Processing train/ai folder..."
mkdir -p "${OUTPUT_DIR}/train/ai"
python ./preprocessing/LNP/test_sidd_rgb_test.py \
    --input_dir "${INPUT_DIR}/train/ai" \
    --result_dir "${OUTPUT_DIR}/train/ai" \
    --weights "${WEIGHTS}" \
    --gpus "6" \
    --save_images

# Process train/real folder
echo "Processing train/real folder..."
mkdir -p "${OUTPUT_DIR}/train/real"
python ./preprocessing/LNP/test_sidd_rgb_test.py \
    --input_dir "${INPUT_DIR}/train/real" \
    --result_dir "${OUTPUT_DIR}/train/real" \
    --weights "${WEIGHTS}" \
    --gpus "6" \
    --save_images

# Process val/ai folder
echo "Processing val/ai folder..."
mkdir -p "${OUTPUT_DIR}/val/ai"
python ./preprocessing/LNP/test_sidd_rgb_test.py \
    --input_dir "${INPUT_DIR}/val/ai" \
    --result_dir "${OUTPUT_DIR}/val/ai" \
    --weights "${WEIGHTS}" \
    --gpus "6" \
    --save_images

# Process val/real folder
echo "Processing val/real folder..."
mkdir -p "${OUTPUT_DIR}/val/real"
python ./preprocessing/LNP/test_sidd_rgb_test.py \
    --input_dir "${INPUT_DIR}/val/real" \
    --result_dir "${OUTPUT_DIR}/val/real" \
    --weights "${WEIGHTS}" \
    --gpus "6" \
    --save_images

echo "Processing complete!" 