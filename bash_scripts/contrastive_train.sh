# PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=/home/czq/workspace/GCD/generalized-category-discovery/outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

python -m methods.contrastive_training.contrastive_training \
            --dataset_name 'imagenet' \
            --setting 'default'\
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 200 \
            --base_model vit_dino \
            --num_workers 8 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v1' 'v2' \
            --use_coarse_label 'False'\
            --eval_freq 200 \
            --sup_coarse_con_weight 1.0 \
 > ${SAVE_DIR}logfile_${EXP_NUM}.out
#  --warmup_model_dir '/home/czq/workspace/GCD/generalized-category-discovery/metric_learn_gcd/log/(17.10.2023_|_33.614)/checkpoints' \
            
 

# EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
# EXP_NUM=$((${EXP_NUM}+1))
# echo $EXP_NUM

# python -m methods.contrastive_training.contrastive_training \
#             --dataset_name 'cifar10' \
#             --setting 'animal_0.5_transportation_0.5_1_1'\
#             --batch_size 128 \
#             --grad_from_block 11 \
#             --epochs 80 \
#             --base_model vit_dino \
#             --num_workers 8 \
#             --use_ssb_splits 'True' \
#             --sup_con_weight 0.35 \
#             --weight_decay 5e-5 \
#             --contrast_unlabel_only 'False' \
#             --transform 'imagenet' \
#             --lr 0.1 \
#             --eval_funcs 'v1' 'v2' \
# > ${SAVE_DIR}logfile_${EXP_NUM}.out