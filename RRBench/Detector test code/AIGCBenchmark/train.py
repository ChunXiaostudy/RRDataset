import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image

from tqdm import tqdm
from validate import validate
from data import create_dataloader_new,create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options import TrainOptions
from data.process import get_processing_model
from util import set_random_seed, get_model
from eval_config import DATASET_PATHS, GPU_MAPPING, WEIGHT_PATHS

import logging
import os
import time

def setup_logging(opt):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_directory = './logs'
    
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_filename = f"finetune_{opt.detect_method}.log"
    
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        filename=os.path.join(log_directory, log_filename),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def get_val_opt(dataset_base):
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = os.path.join(dataset_base, 'val')  # 直接指向val目录
    val_opt.isTrain = False
    val_opt.isVal = True
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_opt

if __name__ == '__main__':
    set_random_seed()
    opt = TrainOptions().parse()
    
    # 设置数据集路径
    if opt.detect_method == 'LNP':
        dataset_base = "/data/lcx/LNP_ORIGINAL_TRAIN_VAL"
    elif opt.detect_method == 'DIRE':
        dataset_base = "/data/lcx/DIRE_ORIGINAL_TRAIN_VAL"
    elif opt.detect_method == 'LGrad':
        dataset_base = "/data/lcx/LGRAD_ORIGINAL_TRAIN_VAL"
    elif opt.detect_method == 'DNF':  # 添加DNF的训练集和验证集路径
        dataset_base = "/data/lcx/DNF_ORIGINAL_TRAIN_VAL"
    else:
        dataset_base = "/data/lcx/RRDataset_original_train_val"
    
    opt.dataroot = os.path.join(dataset_base, 'train')
    
    # 设置设备
    opt.device = GPU_MAPPING[opt.detect_method]
    
    # 设置保存路径
    save_dir = f"/data/lcx/Weight_finetune/{opt.detect_method}"
    os.makedirs(save_dir, exist_ok=True)
    opt.checkpoints_dir = save_dir
    
    # 初始化预处理模型和参数
    opt = get_processing_model(opt)
    
    # 加载预训练模型
    model = get_model(opt)
    state_dict = torch.load(WEIGHT_PATHS[opt.detect_method], map_location='cpu')
    
    if opt.detect_method in ["FreDect", "Gram"]:
        try:
            model.load_state_dict(state_dict['netC'])
        except:
            model.load_state_dict(state_dict['model'])
    elif opt.detect_method == "UnivFD":
        if 'model' in state_dict:
            fc_state_dict = {k.replace("fc.", ""): v for k, v in state_dict['model'].items() if k.startswith("fc.")}
            model.fc.load_state_dict(fc_state_dict)
        else:
            model.fc.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict['model'])
    
    model = model.to(opt.device)
    model.train()
    
    # 设置优化器和学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 设置数据加载器
    train_loader = create_dataloader_new(opt)
    
    # 设置验证数据加载器
    val_opt = get_val_opt(dataset_base)  # 传入dataset_base
    val_opt.device = opt.device
    val_opt = get_processing_model(val_opt)  # 验证集也需要初始化预处理参数
    
    setup_logging(opt)
    
    # 训练循环
    for epoch in range(8):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")
        
        for i, data in enumerate(progress_bar):
            if opt.detect_method == "Fusing":
                input_img = data[0]  # (batch_size, 6, 3, 224, 224)
                cropped_img = data[1].to(opt.device)
                labels = data[2].to(opt.device)
                scale = data[3].to(opt.device)
                outputs = model(input_img, cropped_img, scale)
            else:
                images = data[0].to(opt.device)
                labels = data[1].to(opt.device)
                outputs = model(images)
            
            # 调整标签维度以匹配输出
            labels = labels.view(-1, 1).float()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss/(i+1)})
        
        # 验证
        model.eval()
        acc, ap = validate(model, val_opt)[:2]
        logging.info(f"Epoch {epoch+1}/3 - Val Accuracy: {acc:.4f}, AP: {ap:.4f}")
        
        # 保存模型
        save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, save_path)
        logging.info(f"Model saved to {save_path}")
    
    logging.info("Training completed!")

