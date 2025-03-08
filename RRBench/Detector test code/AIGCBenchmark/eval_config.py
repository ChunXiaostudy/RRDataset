import os

# 数据集根目录配置
DATASET_PATHS = {
    'CNNSpot': '/data/lcx/RRDataset_final',
    'FreDect': '/data/lcx/RRDataset_final', 
    'Fusing': '/data/lcx/RRDataset_final',
    'Gram': '/data/lcx/RRDataset_final',
    'UnivFD': '/data/lcx/RRDataset_final',
    'LNP': '/data/lcx/LNP_IMAGE_TEST',
    'DIRE': '/data/lcx/DIRE_IMAGE_TEST',
    'LGrad': '/data/lcx/LGRAD_IMAGE_TEST',
    'DNF': '/data/lcx/DNF_IMAGE_TEST'
}

# GPU 配置
GPU_MAPPING = {
    'CNNSpot': 'cuda:0',
    'FreDect': 'cuda:1',
    'Fusing': 'cuda:2', 
    'Gram': 'cuda:3',
    'UnivFD': 'cuda:4',
    'LNP': 'cuda:5',
    'DIRE': 'cuda:6',
    'LGrad': 'cuda:7',
    'DNF': 'cuda:0'
}

# 模型权重路径配置 
WEIGHT_PATHS = {
    'CNNSpot': './checkpoints/CNNSpot/model_epoch_best_14.pth',
    'FreDect': './checkpoints/FreDect/fredct14.pth',
    'Fusing': './checkpoints/Fusing/fusing_14.pth',
    'Gram': './checkpoints/Gram/Gram14.pth',
    'UnivFD': '/data/lcx/checkpoints/UnivFD/univfd_14.pth',
    'LNP': '/data/lcx/checkpoints/LNP/CNNSpot/lnp.pth',
    'DIRE': '/data/lcx/checkpoints/DIRE/CNNSpot/dire.pth',
    'LGrad': '/data/lcx/checkpoints/lgrad/CNNSpot/lgard.pth',
    'DNF': '/data/lcx/checkpoints/DNF_v14/CNNSpot/cnnspot.pth'
}

# 测试子集
TEST_SUBSETS = ['original', 'transfer', 'redigital']
# TEST_SUBSETS = ['transfer']

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# root to the testsets
dataroot = '/data'


# list of synthesis algorithms
print(dataroot)
vals = ['progan']

