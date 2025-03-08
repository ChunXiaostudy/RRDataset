'''
python eval_all.py --model_path ./weights/{}.pth --detect_method {CNNSpot,Gram,Fusing,FreDect,LGrad,LNP,DIRE}  --noise_type {blur,jpg,resize}
'''

import os
import csv
import torch
import argparse
from validate import validate
from options import TestOptions
from eval_config import *
from PIL import ImageFile
from util import create_argparser, get_model, set_random_seed

ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_method', type=str, required=True,
                      help='Detection method to use: CNNSpot/FreDect/Fusing/Gram/UnivFD/LNP/DIRE/LGrad')
    return parser.parse_args()

def main():
    args = parse_args()
    set_random_seed()
    
    
    opt = TestOptions().parse(print_options=True)
    opt.detect_method = args.detect_method
    opt.model_path = WEIGHT_PATHS[args.detect_method]
    opt.device = GPU_MAPPING[args.detect_method]
    
    
    opt.blur_prob = 0
    opt.jpg_prob = 0
    opt.no_flip = True
    
    
    results_dir = f"./unfinetue_result_sd14/{opt.detect_method}"
    mkdir(results_dir)
    
    print(f"\nLoading {opt.detect_method} model from {opt.model_path}...")
    
    
    model = get_model(opt)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    
    
    try:
        if opt.detect_method in ["FreDect","Gram"]:
            try:
                model.load_state_dict(state_dict['netC'], strict=True)
            except:
                
                model.load_state_dict(state_dict['model'], strict=True)
        elif opt.detect_method == "UnivFD":
            
            if 'model' in state_dict:
                fc_state_dict = {k.replace("fc.", ""): v for k, v in state_dict['model'].items() if k.startswith("fc.")}
                model.fc.load_state_dict(fc_state_dict)
            else:
                model.fc.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict['model'], strict=True)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {str(e)}")
        return
        
    model = model.to(opt.device)
    model.eval()

    
    rows = [
        [f"{opt.detect_method} model testing results"],
        ['testset', 'accuracy', 'real_acc', 'fake_acc']
    ]

    
    for subset in TEST_SUBSETS:
        print(f"\nTesting on {subset} subset...")
        opt.dataroot = os.path.join(DATASET_PATHS[opt.detect_method], subset)
        acc, ap, r_acc, f_acc, _, _ = validate(model, opt)
        rows.append([subset, acc, r_acc, f_acc])
        print(f"Results for {subset}:")
        print(f"Overall accuracy: {acc:.4f}")
        print(f"Real accuracy: {r_acc:.4f}")
        print(f"Fake accuracy: {f_acc:.4f}")

    
    csv_name = os.path.join(results_dir, f'{opt.detect_method}_results_transfer_1.csv')
    with open(csv_name, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(rows)
    print(f"\nResults saved to {csv_name}")

if __name__ == '__main__':
    main()
