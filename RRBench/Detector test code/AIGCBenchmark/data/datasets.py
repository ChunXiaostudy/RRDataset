import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile

import torchvision
import os
import copy
import torch
from scipy import fftpack
import imageio
from skimage.transform import resize

from .process import *


ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize) # 随机剪裁，默认224
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img) # 不处理
    else:
        crop_func = transforms.CenterCrop(opt.cropSize) # 中心裁剪

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path




rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    width, height = img.size
    # print('before resize: '+str(width)+str(height))
    # quit()
    interp = sample_discrete(opt.rz_interp)
    img = torchvision.transforms.Resize((opt.loadSize,opt.loadSize))(img) 
    return img


def custom_augment(img, opt):
    
    # print('height, width:'+str(height)+str(width))
    # resize
    if opt.noise_type=='resize':
        
        height, width = img.height, img.width
        img = torchvision.transforms.Resize((int(height/2),int(width/2)))(img) 

    img = np.array(img)
    # img = img[0:-1:4,0:-1:4,:]
    if opt.noise_type=='blur':
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if opt.noise_type=='jpg':
        
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)
    
    return Image.fromarray(img)


def loadpathslist(root,flag):
    classes =  os.listdir(root)
    paths = []
    if not '1_fake' in classes:
        for class_name in classes:
            imgpaths = os.listdir(root+'/'+class_name +'/'+flag+'/')
            for imgpath in imgpaths:
                paths.append(root+'/'+class_name +'/'+flag+'/'+imgpath)
        return paths
    else:
        imgpaths = os.listdir(root+'/'+flag+'/')
        for imgpath in imgpaths:
            paths.append(root+'/'+flag+'/'+imgpath)
        return paths

def process_img(img,opt,imgname,target):
    if opt.detect_method in ['CNNSpot','Gram']:
        img = processing(img,opt,'imagenet')
    elif opt.detect_method == 'FreDect':
        img = processing_DCT(img,opt)
    elif opt.detect_method == 'Fusing':
        input_img, cropped_img, scale = processing_PSM(img,opt)
        return input_img, cropped_img, target, scale, imgname
    elif opt.detect_method == 'LGrad':
        opt.cropSize=256
        img = processing_LGrad(img, opt.gen_model, opt)
    elif opt.detect_method == 'LNP':
        img = processing_LNP(img, opt.model_restoration, opt, imgname)
    elif opt.detect_method in ['DIRE', 'DNF']:
        img = processing_DIRE(img,opt,imgname)
    elif opt.detect_method == 'UnivFD':
        img = processing(img, opt,'clip')
    else:
        raise ValueError(f"Unsupported model_type: {opt.detect_method}")

    return img, target
    



class read_data_new():
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        fake_img_list = self.load_image_list(os.path.join(self.root, 'ai'), label=1)
        real_img_list = self.load_image_list(os.path.join(self.root, 'real'), label=0)

        self.imgs = real_img_list + fake_img_list
        self.labels = [0] * len(real_img_list) + [1] * len(fake_img_list)

        # print('directory, realimg, fakeimg:', self.root, len(real_img_list), len(fake_img_list))

    def load_image_list(self, directory, label):
        # 定义图片文件扩展名
        valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        # 只返回图片文件
        return [(os.path.join(directory, fname), label) 
                for fname in os.listdir(directory) 
                if fname.lower().endswith(valid_extensions)]


    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        imgname = self.imgs[index]
        if self.opt.detect_method == 'FreDect':
            img = processing_DCT(img, self.opt)
            

        
        elif self.opt.detect_method in ['CNNSpot','Gram','Steg']:
            img = processing(img,self.opt,'imagenet')
        elif self.opt.detect_method == 'Fusing':
            input_img, cropped_img, scale = processing_PSM(img,self.opt)
            return input_img, cropped_img, target, scale, imgname
        else:
            # 所有其他方法都使用相同的处理方式
            img = processing(img, self.opt, 'imagenet')
        
        return img, target

    def __len__(self):
        return len(self.labels)

# class Option:
#     def __init__(self):
#         self.dataroot = "/data/lcx/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train"  # 根据您的实际路径进行修改
#         self.isTrain = True  # 根据需要选择是训练集还是验证集
#         self.detect_method = 'CNNSpot'
#         self.CropSize = 224
#         self.no_flip = False
#         self.rz_interp = 'bilinear'
#         self.loadSize = 256
#         self.blur_prob = 0
#         self.jpg_prob = 0
    
# if __name__ == "__main__":
#     import torch
#     from torch.utils.data import DataLoader
#     opt = Option()
#     dataset = read_data_new(opt)
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#     for i, (images, labels) in enumerate(dataloader):
#         print("Batch", i + 1)
#         for img, label in zip(images, labels):
#             print(f"Image path: {img}, Label: {label}")
#         break