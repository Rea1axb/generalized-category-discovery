# PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=3

# python -m methods.clustering.extract_features \
#   --dataset cifar10 \
#   --setting 'animal_1.0_transportation_0.0_1_1' \
#   --use_best_model 'False' \
#   --warmup_model_dir './metric_learn_gcd/log/(15.10.2023_|_32.528)/checkpoints/model.pt'

python -m methods.clustering.extract_features \
  --dataset cifar10 \
  --setting 'default' \
  --use_best_model 'False' \
  --warmup_model_dir './metric_learn_gcd/log/(15.10.2023_|_27.077)/checkpoints/model.pt'

python -m methods.clustering.extract_features \
  --dataset cifar100 \
  --setting 'default' \
  --use_best_model 'False' \
  --warmup_model_dir './metric_learn_gcd/log/(16.10.2023_|_39.019)/checkpoints/model.pt'

# python -m methods.clustering.extract_features --dataset cifar100 --use_best_model 'True' \
#  --warmup_model_dir '/work/sagar/osr_novel_categories/metric_learn_gcd/log/(28.04.2022_|_27.530)/checkpoints/model.pt'