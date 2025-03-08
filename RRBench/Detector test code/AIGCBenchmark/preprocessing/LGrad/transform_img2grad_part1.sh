#!/bin/bash

GANmodelpath=$(cd $(dirname $0); pwd)/


INPUT_DIR="/data/RRDataset_original_train_val"
OUTPUT_DIR="/data/LGRAD_ORIGINAL_TRAIN_VAL"


MODEL_WEIGHTS="./weights/preprocessing/karras2019stylegan-bedrooms-256x256_discriminator.pth"


export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=0


echo "Processing train/ai folder..."
mkdir -p "${OUTPUT_DIR}/train/ai"
python $GANmodelpath/gen_imggrad.py \
    "${INPUT_DIR}/train/ai" \
    "${OUTPUT_DIR}/train/ai" \
    "${MODEL_WEIGHTS}" \
    1 \
    resize


echo "Processing train/real folder..."
mkdir -p "${OUTPUT_DIR}/train/real"
python $GANmodelpath/gen_imggrad.py \
    "${INPUT_DIR}/train/real" \
    "${OUTPUT_DIR}/train/real" \
    "${MODEL_WEIGHTS}" \
    1 \
    resize


echo "Processing val/ai folder..."
mkdir -p "${OUTPUT_DIR}/val/ai"
python $GANmodelpath/gen_imggrad.py \
    "${INPUT_DIR}/val/ai" \
    "${OUTPUT_DIR}/val/ai" \
    "${MODEL_WEIGHTS}" \
    1 \
    resize


echo "Processing val/real folder..."
mkdir -p "${OUTPUT_DIR}/val/real"
python $GANmodelpath/gen_imggrad.py \
    "${INPUT_DIR}/val/real" \
    "${OUTPUT_DIR}/val/real" \
    "${MODEL_WEIGHTS}" \
    1 \
    resize

echo "Processing complete!" 