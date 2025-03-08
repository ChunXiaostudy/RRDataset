#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export NCCL_P2P_DISABLE=1

MODEL_PATH="/home/AIGCDetectBenchmark-main/weights/preprocessing/lsun_bedroom.pt"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"


Imgrootdir="RRDataset/test_image/original_image"

Saverootdir="RRDataset/dire_image/original_image"


subdirs=("nature" "ai")
for subdir in "${subdirs[@]}"
do
    images_dir="${Imgrootdir}/${subdir}"
    recons_dir="${Saverootdir}/${subdir}"
    dire_dir="${Saverootdir}/${subdir}"


    num_images=$(find "${images_dir}" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.JPEG" -o -iname "*.PNG" \) | wc -l)
    
    SAVE_FLAGS="--images_dir $images_dir --recons_dir $recons_dir --dire_dir $dire_dir"
    SAMPLE_FLAGS="--batch_size 8 --num_samples $num_images --timestep_respacing ddim20 --use_ddim True"
    
    mpiexec -n 1 python /home/AIGCDetectBenchmark-main/preprocessing/DIRE/compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder True --has_subclasses False
done
