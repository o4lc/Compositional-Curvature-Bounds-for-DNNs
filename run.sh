#!/bin/bash

gpu=0

dataset=cifar10
model_name=3C1F
name=tanhTest
dir="$dataset/$model_name$name"
activation=tanh

CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset $dataset\
                --model-name $model_name --train_dir $dir --epochs 1000 --batch_size 256 --activation $activation\
                --scheduler cosine --lr 0.005 --save_checkpoint_epochs 5 --cpl\
                # --penalizeCurvature --hessianRegularizerCoefficient 0.1 --hessianRegularizerPrimalDualStepSize 0.01\
                # --hessianRegularizerPrimalDualEpsilon 0.5 --hessianRegularizerMinimumCoefficient 0.001 \


# CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode certified_attack --dataset $dataset --model-name $model_name\
            #  --train_dir $dir --activation $activation

#CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode attack --dataset $dataset --model-name $model_name\
#             --train_dir $dir --activation $activation
#
#CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode certified --dataset $dataset --model-name $model_name\
#             --train_dir $dir --activation $activation --newtonStep