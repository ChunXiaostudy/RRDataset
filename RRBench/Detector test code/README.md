# AI-Generated Image Detection Test Framework

This repository contains test code for evaluating various AI-generated image detection methods. The code is adapted and modified from several state-of-the-art detection methods to create a unified testing framework.

## Overview

This test framework is built upon the following repositories:
- [AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark)
- [FreqNet-DeepfakeDetection](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection)
- [DRCT](https://github.com/beibuwandeluori/DRCT)
- [C2P-CLIP-DeepfakeDetection](https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection)
- [SAFE](https://github.com/Ouxiang-Li/SAFE)
- [AIDE](https://github.com/shilinyan99/AIDE)

The code has been modified to adapt to our dataset structure while maintaining the original functionality and testing procedures.

## Repository Structure

```
.
├── AIGCBenchmark/        # Main benchmark testing framework
├── C2P-clip/            # C2P-CLIP testing code
├── DRCT/                # DRCT testing code
├── SAFE/                # SAFE testing code
├── AIDE/                # AIDE testing code
└── FreqNet&NPR/         # FreqNet testing code
```

## Key Features

- Unified testing framework for multiple detection methods
- Consistent dataset interface
- Standardized evaluation metrics
- Added support for DNF method in AIGCBenchmark
- Preserved original training and testing procedures

## Environment Setup

Please follow the environment setup instructions from the original repositories. Each method maintains its original dependencies and requirements.

## Usage

### AIGCBenchmark Testing

```bash
python eval_all.py --model_path ./weights/[METHOD_NAME].pth --detect_method [METHOD_NAME] --no_resize --no_crop --batch_size 1
```

### C2P-CLIP Testing

```bash
python inference.py --dataroot [YOUR_DATA_PATH] --model_path [MODEL_PATH] --save_path ./results.csv
```

### SAFE Testing

```bash
sh scripts/eval.sh
```

### AIDE Testing

```bash
python main_finetune.py --model AIDE --batch_size 16 --eval True --device cuda:0 --resume [MODEL_PATH] --eval_data_path [DATA_PATH]
```

## Dataset Structure

The test framework expects the following dataset structure:

```
dataset_root/
├── original/
│   ├── real_images/
│   └── ai_images/
├── transfer/
│   ├── real_images/
│   └── ai_images/
└── redigital/
    ├── real_images/
    └── ai_images/
```

## Notes

- All methods maintain their original training and testing procedures
- The framework provides a unified interface for evaluation
- Results are saved in a standardized format for easy comparison
- The code supports various preprocessing methods (LNP, DIRE, LGrad, DNF)

## Acknowledgments

This test framework is built upon the excellent work of the original authors of the detection methods. We thank them for making their code publicly available.