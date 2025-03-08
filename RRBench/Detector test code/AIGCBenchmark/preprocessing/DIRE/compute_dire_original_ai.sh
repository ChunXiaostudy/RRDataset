#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
export NCCL_P2P_DISABLE=1

MODEL_PATH="/home/AIGCDetectBenchmark-main/weights/preprocessing/lsun_bedroom.pt"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

# Input and output directories
INPUT_DIR="/data//RRDataset_original_train_val"
OUTPUT_DIR="/data//DIRE_ORIGINAL_TRAIN_VAL"

# Process train/ai folder
echo "Processing train/ai folder..."
num_images=$(find "${INPUT_DIR}/train/ai" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.JPEG" -o -iname "*.PNG" \) | wc -l)
SAVE_FLAGS="--images_dir ${INPUT_DIR}/train/ai --recons_dir ${OUTPUT_DIR}/train/ai --dire_dir ${OUTPUT_DIR}/train/ai"
SAMPLE_FLAGS="--batch_size 8 --num_samples $num_images --timestep_respacing ddim20 --use_ddim True"
mkdir -p "${OUTPUT_DIR}/train/ai"
mpiexec -n 1 python /home/AIGCDetectBenchmark-main/preprocessing/DIRE/compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder False --has_subclasses False



# Process val/real folder
echo "Processing val/real folder..."
num_images=$(find "${INPUT_DIR}/val/real" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.JPEG" -o -iname "*.PNG" \) | wc -l)
SAVE_FLAGS="--images_dir ${INPUT_DIR}/val/real --recons_dir ${OUTPUT_DIR}/val/real --dire_dir ${OUTPUT_DIR}/val/real"
SAMPLE_FLAGS="--batch_size 8 --num_samples $num_images --timestep_respacing ddim20 --use_ddim True"
mkdir -p "${OUTPUT_DIR}/val/real"
mpiexec -n 1 python /home/AIGCDetectBenchmark-main/preprocessing/DIRE/compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder False --has_subclasses False 