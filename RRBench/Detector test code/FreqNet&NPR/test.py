import sys
import time
import os
import torch
from util import Logger, printSet
from validate import validate
from networks.freqnet import freqnet
from options.test_options import TestOptions
import numpy as np
from tqdm import tqdm


DetectionTests = {
    'original': {
        'dataroot': '/data/RRDataset_final/original',
        'no_resize': False,
        'no_crop': True,
    },
    'transfer': {
        'dataroot': '/data/RRDataset_final/transfer', 
        'no_resize': False,
        'no_crop': True,
    },
    'redigital': {
        'dataroot': '/data/RRDataset_final/redigital',
        'no_resize': False,
        'no_crop': True,
    }
}


result_dir = '/data/Weight_finetune/Freq/result'
os.makedirs(result_dir, exist_ok=True)


Logger(os.path.join(result_dir, 'test_log.txt'))

opt = TestOptions().parse(print_options=False)
opt.model_path = '/data/Weight_finetune/Freq/model_epoch_last.pth'
print(f'Model_path: {opt.model_path}')


model = freqnet(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.cuda()
model.eval()


results = []


for testSet in tqdm(DetectionTests.keys(), desc="Testing datasets"):
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)
    
    print(f"\nTesting {testSet}...")
    opt.dataroot = dataroot
    opt.no_resize = DetectionTests[testSet]['no_resize'] 
    opt.no_crop = DetectionTests[testSet]['no_crop']
    
    acc, ap, r_acc, f_acc, _, _ = validate(model, opt)
    
    
    result = {
        'dataset': testSet,
        'accuracy': acc * 100,
        'real_accuracy': r_acc * 100,
        'fake_accuracy': f_acc * 100,
        'ap': ap * 100
    }
    results.append(result)
    
    
    print(f"\n{testSet} Results:")
    print(f"ACC: {acc*100:.2f}%")
    print(f"REAL_ACC: {r_acc*100:.2f}%")
    print(f"FAKE_ACC: {f_acc*100:.2f}%")
    print(f"AP: {ap*100:.2f}%")
    print("-"*50)


with open(os.path.join(result_dir, 'test_results.txt'), 'w') as f:
    f.write("Test Results Summary\n")
    f.write("="*50 + "\n\n")
    
    for result in results:
        f.write(f"Dataset: {result['dataset']}\n")
        f.write(f"Overall Accuracy: {result['accuracy']:.2f}%\n")
        f.write(f"Real Image Accuracy: {result['real_accuracy']:.2f}%\n")
        f.write(f"Fake Image Accuracy: {result['fake_accuracy']:.2f}%\n")
        f.write(f"Average Precision: {result['ap']:.2f}%\n")
        f.write("-"*30 + "\n\n")

print(f"\nResults have been saved to {result_dir}")

