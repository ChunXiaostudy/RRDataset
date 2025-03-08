import os


DATASET_PATHS = {
    'CNNSpot': '/data/RRDataset_final',
    'FreDect': '/data/RRDataset_final', 
    'Fusing': '/data/RRDataset_final',
    'Gram': '/data/RRDataset_final',
    'UnivFD': '/data/RRDataset_final',
    'LNP': '/data/LNP_IMAGE_TEST',
    'DIRE': '/data/DIRE_IMAGE_TEST',
    'LGrad': '/data/LGRAD_IMAGE_TEST',
    'DNF': '/data/DNF_IMAGE_TEST'
}


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


WEIGHT_PATHS = {
    'CNNSpot': './checkpoints/CNNSpot/model_epoch_best_14.pth',
    'FreDect': './checkpoints/FreDect/fredct14.pth',
    'Fusing': './checkpoints/Fusing/fusing_14.pth',
    'Gram': './checkpoints/Gram/Gram14.pth',
    'UnivFD': './checkpoints/UnivFD/univfd_14.pth',
    'LNP': './checkpoints/LNP/CNNSpot/lnp.pth',
    'DIRE': './checkpoints/DIRE/CNNSpot/dire.pth',
    'LGrad': './checkpoints/lgrad/CNNSpot/lgard.pth',
    'DNF': './checkpoints/DNF_v14/CNNSpot/cnnspot.pth'
}


TEST_SUBSETS = ['original', 'transfer', 'redigital']


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# root to the testsets
dataroot = '/data'


# list of synthesis algorithms
print(dataroot)
vals = ['progan']

