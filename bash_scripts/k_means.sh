# PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=3

# Get unique log file
SAVE_DIR=/home/czq/workspace/GCD/generalized-category-discovery/outputs/



# python -m methods.clustering.k_means --dataset 'cifar100' --semi_sup 'True' --use_ssb_splits 'True' \
#  --max_kmeans_iter 200 --k_means_init 100 \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out

for s in {5..5}
do
EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM
python -m methods.clustering.k_means --dataset 'imagenet' --setting 'default' --semi_sup 'True' \
 --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(09.06.2024_|_22.289)' --use_coarse_label 'False' --seed $s \
 > ${SAVE_DIR}logfile_${EXP_NUM}.out
done

# for s in {4..5}
# do
# EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
# EXP_NUM=$((${EXP_NUM}+1))
# echo $EXP_NUM
# python -m methods.clustering.k_means --dataset 'cifar100' --setting 'default' --semi_sup 'True' \
#  --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(11.04.2024_|_08.020)' --use_coarse_label 'False' --seed $s \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out
# done

# for s in {4..5}
# do
# EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
# EXP_NUM=$((${EXP_NUM}+1))
# echo $EXP_NUM
# python -m methods.clustering.k_means --dataset 'cub' --setting 'default' --semi_sup 'True' \
#  --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(13.04.2024_|_17.257)' --use_coarse_label 'False' --seed $s \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out
# done

# for s in {4..5}
# do
# EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
# EXP_NUM=$((${EXP_NUM}+1))
# echo $EXP_NUM
# python -m methods.clustering.k_means --dataset 'scars' --setting 'default' --semi_sup 'True' \
#  --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(13.04.2024_|_29.051)' --use_coarse_label 'False' --seed $s \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out
# done

# for s in {4..5}
# do
# EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
# EXP_NUM=$((${EXP_NUM}+1))
# echo $EXP_NUM
# python -m methods.clustering.k_means --dataset 'aircraft' --setting 'default' --semi_sup 'True' \
#  --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(13.04.2024_|_39.011)' --use_coarse_label 'False' --seed $s \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out
# done
# python -m methods.clustering.k_means --dataset 'cifar100' --setting 'default' --semi_sup 'True' \
#  --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(16.10.2023_|_39.019)' \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out

#  python -m methods.clustering.k_means --dataset 'cifar10' --setting 'animal_2_transportation_0_0.5' --semi_sup 'True' \
#  --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(24.10.2023_|_23.769)' \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out

#  python -m methods.clustering.k_means --dataset 'cifar100' --setting 'completely_old' --semi_sup 'True' \
#  --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(29.10.2023_|_38.584)' \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out