# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '../../../CIFAR10'
cifar_100_root = '../../../CIFAR100'
# cub_root = '/work/sagar/datasets/CUB'
# aircraft_root = '/work/khan/datasets/aircraft/fgvc-aircraft-2013b'
# herbarium_dataroot = '/work/sagar/datasets/herbarium_19/'
# imagenet_root = '/scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12'

cub_root = '../../../data/CUB'
aircraft_root = '../../../data/fgvc-aircraft-2013b'
car_root = '../../../data/stanford_cars'
herbarium_dataroot = '../../../data/herbarium_19'
# imagenet_root = '../../../data/ImageNet'
imagenet_root = '../../../data/imagenet100_small'
imagenet_200_root = '../../../data/imagenet200_small'

# OSR Split dir
# osr_split_dir = '/users/sagar/kai_collab/osr_novel_categories/data/ssb_splits'
osr_split_dir = './data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = './pretrained_models/dino_vitbase16_pretrain.pth'
feature_extract_dir = './extracted_features'     # Extract features to this directory
exp_root = './'          # All logs and checkpoints will be saved here