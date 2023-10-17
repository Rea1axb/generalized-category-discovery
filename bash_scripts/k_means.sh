# PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=3

# Get unique log file
SAVE_DIR=/home/czq/workspace/GCD/generalized-category-discovery/outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

# python -m methods.clustering.k_means --dataset 'cifar100' --semi_sup 'True' --use_ssb_splits 'True' \
#  --max_kmeans_iter 200 --k_means_init 100 \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out

# python -m methods.clustering.k_means --dataset 'cifar100' --setting 'default' --semi_sup 'True' \
#  --use_best_model 'True' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(16.10.2023_|_39.019)' \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out

# python -m methods.clustering.k_means --dataset 'cifar100' --setting 'default' --semi_sup 'True' \
#  --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(16.10.2023_|_39.019)' \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out

#  python -m methods.clustering.k_means --dataset 'cifar10' --setting 'default' --semi_sup 'True' \
#  --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(15.10.2023_|_27.077)' \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out

 python -m methods.clustering.k_means --dataset 'cifar10' --setting 'animal_1.0_transportation_0.0_1_1' --semi_sup 'True' \
 --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(15.10.2023_|_32.528)' \
 > ${SAVE_DIR}logfile_${EXP_NUM}.out