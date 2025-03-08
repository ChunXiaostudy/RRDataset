MODEL="SAFE"
OUTPUT_PATH="/data/MODEL_WEIGHT_FINTUNED/SAFE"
mkdir -p $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
    --input_size 256 \
    --transform_mode 'crop' \
    --model $MODEL \
    --data_path "/data/RRDataset_train/train" \
    --eval_data_path "/data/RRDataset_train/val" \
    --save_ckpt_freq 1 \
    --batch_size 32 \
    --blr 1e-3 \
    --weight_decay 0.01 \
    --warmup_epochs 0 \
    --epochs 3 \
    --num_workers 4 \
    --output_dir $OUTPUT_PATH \
    --resume "./checkpoint/checkpoint-best.pth" \
    --dist_url "none" \
    2>&1 | tee -a $OUTPUT_PATH/log_train.txt