MODEL="SAFE"
RESUME_PATH="/data/zhuyao/MODEL_WEIGHT_FINTUNED/SAFE/checkpoint-best.pth"
OUTPUT_PATH="/data/zhuyao/MODEL_WEIGHT_FINTUNED/SAFE"

# 创建输出目录
mkdir -p $OUTPUT_PATH

# 评估original数据集
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
    --input_size 256 \
    --transform_mode 'crop' \
    --model $MODEL \
    --eval_data_path "/data/zhuyao/lcx/RRDataset_final/original" \
    --batch_size 32 \
    --num_workers 4 \
    --output_dir $OUTPUT_PATH \
    --resume $RESUME_PATH \
    --eval True \
    --dist_url "none" \
    2>&1 | tee $OUTPUT_PATH/log_eval_original.txt

# 评估transfer数据集
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
    --input_size 256 \
    --transform_mode 'crop' \
    --model $MODEL \
    --eval_data_path "/data/zhuyao/lcx/RRDataset_final/transfer" \
    --batch_size 32 \
    --num_workers 4 \
    --output_dir $OUTPUT_PATH \
    --resume $RESUME_PATH \
    --eval True \
    --dist_url "none" \
    2>&1 | tee $OUTPUT_PATH/log_eval_transfer.txt

# 评估redigital数据集
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
    --input_size 256 \
    --transform_mode 'crop' \
    --model $MODEL \
    --eval_data_path "/data/zhuyao/lcx/RRDataset_final/redigital" \
    --batch_size 32 \
    --num_workers 4 \
    --output_dir $OUTPUT_PATH \
    --resume $RESUME_PATH \
    --eval True \
    --dist_url "none" \
    2>&1 | tee $OUTPUT_PATH/log_eval_redigital.txt

# 合并结果到CSV
python - <<EOF
import os
import re
import csv

def extract_metrics(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
        acc = re.search(r"Accuracy of the network.*?: ([\d.]+%)", content).group(1)
        real_acc = re.search(r"Real Accuracy: ([\d.]+%)", content).group(1)
        fake_acc = re.search(r"Fake Accuracy: ([\d.]+%)", content).group(1)
        ap = re.search(r"AP: ([\d.]+%)", content).group(1)
        return [acc, real_acc, fake_acc, ap]

output_dir = "$OUTPUT_PATH"
results = []
for dataset in ['original', 'transfer', 'redigital']:
    log_file = os.path.join(output_dir, f'log_eval_{dataset}.txt')
    metrics = extract_metrics(log_file)
    results.append([dataset] + metrics)

csv_path = os.path.join(output_dir, 'evaluation_results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'ACC', 'REAL_ACC', 'FAKE_ACC', 'AP'])
    writer.writerows(results)

print(f"\nResults saved to {csv_path}")
EOF