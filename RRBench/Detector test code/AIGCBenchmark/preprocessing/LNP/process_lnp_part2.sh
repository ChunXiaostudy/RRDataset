#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

# Model weights path
WEIGHTS="./weights/preprocessing/sidd_rgb.pth"

# Base directories
INPUT_BASE="/data/lcx/RRDataset_final"
OUTPUT_BASE="/data/lcx/LNP_IMAGE"

# Folders to process
FOLDERS=(
    "redigital_real_images"
    "transfer_ai_images"
    "transfer_real_images"
)

# Process each folder
for folder in "${FOLDERS[@]}"; do
    INPUT_DIR="${INPUT_BASE}/${folder}"
    OUTPUT_DIR="${OUTPUT_BASE}/${folder}"
    
    echo "Processing ${folder}..."
    echo "Input: ${INPUT_DIR}"
    echo "Output: ${OUTPUT_DIR}"
    
    # Create output directory if it doesn't exist
    mkdir -p "${OUTPUT_DIR}"
    
    # Run the processing script
    python ./preprocessing/LNP/test_sidd_rgb_test.py \
        --input_dir "${INPUT_DIR}" \
        --result_dir "${OUTPUT_DIR}" \
        --weights "${WEIGHTS}" \
        --gpus "7" \
        --save_images
done

echo "Processing complete for part 2!" 