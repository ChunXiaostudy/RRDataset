import argparse
import warnings
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score
import csv
import glob
import cv2

warnings.filterwarnings("ignore")

from network.models import get_models
from data.dataset import AIGCDetectionDataset
from data.transform import create_val_transforms

def get_parser():
    parser = argparse.ArgumentParser(description="RRDataset Testing")
    parser.add_argument("--model_name", default='convnext_base_in22k', type=str)
    parser.add_argument("--model_path", default='/data/DRCT_finetue/DRCT_2m_final_model_acc0.9560.pth', type=str)  # drct+finetune 
    parser.add_argument("--root_path", default='/data/RRDataset_test_DRCT', type=str)
    parser.add_argument("--device_id", default='2', type=str)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--embedding_size", default=1024, type=int)
    return parser.parse_args()

def find_images(dir_path):
    image_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
      
        real_path = os.path.join(root_dir, 'real')
        if os.path.exists(real_path):
            real_images = find_images(real_path)
            self.images.extend(real_images)
            self.labels.extend([0] * len(real_images))
        
       
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

def evaluate_condition(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.cuda()
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1)
            preds = (preds[:, 1] > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    
    total_acc = accuracy_score(all_labels, all_preds)
    
   
    real_indices = [i for i, label in enumerate(all_labels) if label == 0]
    fake_indices = [i for i, label in enumerate(all_labels) if label == 1]
    
    real_acc = accuracy_score([all_labels[i] for i in real_indices], 
                            [all_preds[i] for i in real_indices]) if real_indices else 0
    fake_acc = accuracy_score([all_labels[i] for i in fake_indices], 
                            [all_preds[i] for i in fake_indices]) if fake_indices else 0
    
    return total_acc, real_acc, fake_acc

def main():
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    
  
    model = get_models(model_name=args.model_name, num_classes=2, embedding_size=args.embedding_size)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model = model.cuda()
    model.eval()
    
   
    conditions = ['original', 'transfer', 'redigital']
    results = []
    
    transform = create_val_transforms(size=args.input_size)
    
  
    for condition in conditions:
        print(f"\nEvaluating condition: {condition}")
        
      
        condition_path = os.path.join(args.root_path, condition)
    
        dataset = CustomDataset(
            root_dir=condition_path,
            transform=transform
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
       
        total_acc, real_acc, fake_acc = evaluate_condition(model, data_loader)
        
        
        results.append({
            'Condition': condition,
            'Total_Accuracy': f"{total_acc:.4f}",
            'Real_Accuracy': f"{real_acc:.4f}",
            'Fake_Accuracy': f"{fake_acc:.4f}"
        })
        
        print(f"{condition} Results:")
        print(f"Total Accuracy: {total_acc:.4f}")
        print(f"Real Accuracy: {real_acc:.4f}")
        print(f"Fake Accuracy: {fake_acc:.4f}")
    

    csv_path = os.path.join('/home/DRCT-main', 'dcrt_fintune_results.csv')
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == '__main__':
    main()