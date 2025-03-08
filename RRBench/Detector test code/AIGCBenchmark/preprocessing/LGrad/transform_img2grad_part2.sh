#!/bin/bash

GANmodelpath=$(cd $(dirname $0); pwd)/

# 输入和输出目录设置
INPUT_BASE="/data/lcx/RRDataset_final"
OUTPUT_BASE="/data/lcx/LGRAD_IMAGE"

# 要处理的文件夹列表
FOLDERS=(
    "redigital_real_images"
    "transfer_ai_images"
    "transfer_real_images"
)

# 模型权重路径
MODEL_WEIGHTS="./weights/preprocessing/karras2019stylegan-bedrooms-256x256_discriminator.pth"

# 修改CUDA设备设置
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1

# 处理每个文件夹
for folder in "${FOLDERS[@]}"; do
    INPUT_DIR="${INPUT_BASE}/${folder}"
    OUTPUT_DIR="${OUTPUT_BASE}/${folder}"
    
    echo "Processing ${folder}..."
    echo "Input: ${INPUT_DIR}"
    echo "Output: ${OUTPUT_DIR}"
    
    # 创建输出目录
    mkdir -p "${OUTPUT_DIR}"
    
    # 运行处理脚本
    python $GANmodelpath/gen_imggrad.py \
        "${INPUT_DIR}" \
        "${OUTPUT_DIR}" \
        "${MODEL_WEIGHTS}" \
        1 \
        resize
done

echo "Part 2 processing complete!" 