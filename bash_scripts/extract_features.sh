# PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# python -m methods.clustering.extract_features \
#   --dataset cifar10 \
#   --setting 'animal_2_transportation_0_0.5' \
#   --use_best_model 'False' \
#   --warmup_model_dir './metric_learn_gcd/log/(24.10.2023_|_23.769)/checkpoints/model.pt'


# extract block features
python -m methods.clustering.extract_features \
  --dataset imagenet \
  --setting 'default' \
  --use_best_model 'False' \
  --warmup_model_dir '/home/czq/workspace/GCD/generalized-category-discovery/metric_learn_gcd/log/(09.06.2024_|_22.289)/checkpoints/model.pt' \
  --extract_block 'False'

# python -m methods.clustering.extract_features \
#   --dataset cifar100 \
#   --setting 'completely_old' \
#   --use_best_model 'False' \
#   --warmup_model_dir './metric_learn_gcd/log/(29.10.2023_|_38.584)/checkpoints/model.pt'

# python -m methods.clustering.extract_features --dataset cifar100 --use_best_model 'True' \
#  --warmup_model_dir '/work/sagar/osr_novel_categories/metric_learn_gcd/log/(28.04.2022_|_27.530)/checkpoints/model.pt'