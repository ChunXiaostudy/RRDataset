import argparse
import warnings
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score
import pytorch_warmup as warmup

from network.models import get_models
from data.transform import create_train_transforms, create_val_transforms

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_parser():
    parser = argparse.ArgumentParser(description="RRDataset Finetuning")
    parser.add_argument("--model_name", default='convnext_base_in22k', type=str)
    parser.add_argument("--pretrained_path", 
                        default='/data/pretrain_weight/pretrained/GenImage/sdv14/convnext_base_in22k_224_drct_amp_crop/last_acc0.9991.pth',
                        type=str)
    parser.add_argument("--save_dir", default='/data/DRCT_finetue', type=str)
    parser.add_argument("--train_root", default='/data/RRDataset_train_DRCT', type=str)
    parser.add_argument("--device_id", default='0', type=str)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--embedding_size", default=1024, type=int)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--is_amp", action='store_true', help='Whether to use mixed precision training')
    return parser.parse_args()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 加载real图片
        real_path = os.path.join(root_dir, 'real')
        if os.path.exists(real_path):
            real_images = find_images(real_path)
            self.images.extend(real_images)
            self.labels.extend([0] * len(real_images))
        
        # 加载ai图片
        ai_path = os.path.join(root_dir, 'ai')
        if os.path.exists(ai_path):
            ai_images = find_images(ai_path)
            self.images.extend(ai_images)
            self.labels.extend([1] * len(ai_images))
        
        print(f"Found {len(self.images)} images in {root_dir}")
        print(f"Real: {len([l for l in self.labels if l == 0])}, Fake: {len([l for l in self.labels if l == 1])}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, self.labels[idx]

def find_images(dir_path):
    image_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)

def evaluate(model, val_loader):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            preds = torch.softmax(outputs, dim=1)
            preds = (preds[:, 1] > 0.5).float()
            acc = (preds == labels.float()).float().mean()
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))
    
    return losses.avg, accuracies.avg

def train_epoch(model, train_loader, criterion, optimizer, scaler=None):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    train_process = tqdm(train_loader, desc="Training")
    for images, labels in train_process:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        preds = torch.softmax(outputs, dim=1)
        preds = (preds[:, 1] > 0.5).float()
        acc = (preds == labels.float()).float().mean()
        
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc.item(), images.size(0))
        
        train_process.set_postfix({
            'Loss': f"{losses.avg:.4f}",
            'Acc': f"{accuracies.avg:.4f}"
        })
    
    return losses.avg, accuracies.avg

def main():
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载预训练模型
    model = get_models(model_name=args.model_name, num_classes=2, embedding_size=args.embedding_size)
    model.load_state_dict(torch.load(args.pretrained_path, map_location='cpu'))
    model = model.cuda()
    
    # 准备数据集
    train_transform = create_train_transforms(size=args.input_size)
    val_transform = create_val_transforms(size=args.input_size)
    
    train_dataset = CustomDataset(
        root_dir=os.path.join(args.train_root, 'train'),
        transform=train_transform
    )
    
    val_dataset = CustomDataset(
        root_dir=os.path.join(args.train_root, 'val'),
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 准备训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = GradScaler() if args.is_amp else None
    
    best_acc = 0.0
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader)
        
        # 更新学习率
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, f'best_model_acc{val_acc:.4f}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
    
    # 保存最后一个epoch的模型
    final_save_path = os.path.join(args.save_dir, f'final_model_acc{val_acc:.4f}.pth')
    torch.save(model.state_dict(), final_save_path)
    print(f"Saved final model to {final_save_path}")

if __name__ == '__main__':
    main() 