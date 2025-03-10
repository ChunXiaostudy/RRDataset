import argparse
import sys
import time
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from data import create_dataloader
import random
from transformers import CLIPModel
import warnings
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(123)

class C2P_CLIP(nn.Module):
    def __init__(self, name='openai/clip-vit-large-patch14', num_classes=1):
        super(C2P_CLIP, self).__init__()
        self.model        = CLIPModel.from_pretrained(name)
        del self.model.text_model
        del self.model.text_projection
        del self.model.logit_scale
        
        self.model.vision_model.requires_grad_(False)
        self.model.visual_projection.requires_grad_(False)
        self.model.fc = nn.Linear( 768, num_classes )
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)

    def encode_image(self, img):
        vision_outputs = self.model.vision_model(
            pixel_values=img,
            output_attentions    = self.model.config.output_attentions,
            output_hidden_states = self.model.config.output_hidden_states,
            return_dict          = self.model.config.use_return_dict,      
        )
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.model.visual_projection(pooled_output)
        return image_features    

    def forward(self, img):
        # tmp = x; print(f'x: {tmp.shape}, max: {tmp.max()}, min: {tmp.min()}, mean: {tmp.mean()}')
        image_embeds = self.encode_image(img)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return self.model.fc(image_embeds)
        
def printSet(set_str):
    set_str = str(set_str)
    num = len(set_str)
    print("="*num*3)
    print(" "*num + set_str)
    print("="*num*3)
    
def parse_args():
    parser = argparse.ArgumentParser(description='test C2P-CLIP')
    parser.add_argument('--loadSize'     ,  type=int  , default=224                                                            )
    parser.add_argument('--cropSize'     ,  type=int  , default=224                                                            )
    parser.add_argument('--batch_size'   ,  type=int  , default=64                                                             )
    parser.add_argument('--dataroot'     ,  type=str  , default='/data/RRDataset_final'                            )
    parser.add_argument('--model_path'   ,  type=str  , default='https://www.now61.com/f/95OefW/C2P_CLIP_release_20240901.zip' )
    parser.add_argument('--save_path'    ,  type=str  , default='./results.csv' )
    args = parser.parse_args()
    
    def print_options(parser, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    print_options(parser, args)
    return args

if __name__ == '__main__':
    opt = parse_args()
    
    
    results = []
    
    
    state_dict = torch.hub.load_state_dict_from_url(opt.model_path, map_location="cpu", progress=True)
    model = C2P_CLIP(name='openai/clip-vit-large-patch14', num_classes=1)
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()

    
    subdatasets = ['original', 'transfer', 'redigital']
    base_dataroot = opt.dataroot  
    
    for subdataset in tqdm(subdatasets, desc="处理数据集"):
        
        dataroot = os.path.join(base_dataroot, subdataset)
        print(f"\nTesting on {subdataset} dataset:")
        
        
        opt.dataroot = dataroot
        data_loader = create_dataloader(opt)
        
        with torch.no_grad():
            y_true, y_pred = [], []
            for img, label, path in tqdm(data_loader, desc=f"推理 {subdataset} 数据集"):
                y_pred.extend(model(img.cuda()).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())
                
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        
        real_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
        fake_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
        total_acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        
        
        results.append({
            'Dataset': subdataset,
            'Total_Accuracy': total_acc * 100,
            'Real_Accuracy': real_acc * 100,
            'Fake_Accuracy': fake_acc * 100,
            'AP': ap * 100
        })
        
        print(f"Dataset: {subdataset}")
        print(f"Total Accuracy: {total_acc*100:.2f}%")
        print(f"Real Accuracy: {real_acc*100:.2f}%")
        print(f"Fake Accuracy: {fake_acc*100:.2f}%")
        print(f"AP: {ap*100:.2f}%")
    
    
    df = pd.DataFrame(results)
    df.to_csv(opt.save_path, index=False)
    print(f"\nResults saved to {opt.save_path}")

