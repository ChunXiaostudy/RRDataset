# GPU_NUM=8
# WORLD_SIZE=1
# RANK=0
# MASTER_ADDR=localhost
# MASTER_PORT=29572

# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPU_NUM \
#     --nnodes $WORLD_SIZE \
#     --node_rank $RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "

PY_ARGS=${@:1}  # Any other arguments 

python main_finetune.py \
    --model AIDE \
    --batch_size 16 \
    --eval True \
    --device cuda:0 \
    --resume ./weights/weight/sd14_train_aide.pth \
    --eval_data_path /data/RRDataset_final \
    --output_dir /data/Weight_finetune/AIDE/unfinetune_result/sd14 \
    ${PY_ARGS} 
