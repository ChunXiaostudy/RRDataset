"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import sys
import os


script_dir = os.path.dirname(os.path.abspath(__file__))


project_root = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(project_root)

import numpy as np
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import cv2
from skimage import img_as_ubyte
from networks.denoising_rgb import DenoiseNet
from preprocessing_model.LNP_model.dataloaders.data_rgb import get_validation_data
import util

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='./test_image',
    help='Directory path to the inputs.')
parser.add_argument('--result_dir', default='./lnp_image',
    help='Directory path to the results.')
parser.add_argument('--weights', default='./weights/preprocessing/sidd_rgb.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='5', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--noise_type', default=None, type=str, help='e.g. jpg, blur, resize')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

util.mkdir(args.result_dir)

def get_image_paths(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.PNG', '.JPEG')):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def main():
    
    image_paths = get_image_paths(args.input_dir)
    print(f"Found {len(image_paths)} images in {args.input_dir}")

    
    test_dataset = get_validation_data(image_paths, args.noise_type)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    
    model_restoration = DenoiseNet()
    util.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_noisy = data_test[0].cuda()
            filenames = data_test[1]
            
            try:
                rgb_restored = model_restoration(rgb_noisy)
                rgb_restored = torch.clamp(rgb_restored, 0, 1)

                rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
                rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                for batch in range(len(rgb_noisy)):
                    denoised_img = img_as_ubyte(rgb_restored[batch])
                    imgsavepath = filenames[batch].replace(args.input_dir, args.result_dir)
                    
                    
                    imgsavepath = imgsavepath.replace('jpg', 'png').replace('JPEG', 'png').replace('jpeg', 'png')
                    
                    try:
                        rootpath = os.path.dirname(imgsavepath)
                        os.makedirs(rootpath, exist_ok=True)
                        cv2.imwrite(imgsavepath, denoised_img * 255)
                    except Exception as e:
                        print(f"Error saving {imgsavepath}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing {filenames}: {str(e)}")
                continue

if __name__ == '__main__':
    main()

                    

