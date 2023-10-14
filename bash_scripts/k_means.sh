# PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=3

# Get unique log file
SAVE_DIR=/home/czq/workspace/GCD/generalized-category-discovery/outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

python -m methods.clustering.k_means --dataset 'cifar100' --semi_sup 'True' --use_ssb_splits 'True' \
 --max_kmeans_iter 200 --k_means_init 100 \
 > ${SAVE_DIR}logfile_${EXP_NUM}.out

# python -m methods.clustering.k_means --dataset 'scars' --semi_sup 'True' --use_ssb_splits 'True' \
#  --use_best_model 'True' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(28.04.2022_|_27.516)' \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out