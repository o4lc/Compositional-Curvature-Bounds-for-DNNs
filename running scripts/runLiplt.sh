#!/bin/bash



gpu=0
dataset=cifar10  # mnist, cifar10
model_name=liplt-6FCifar
name=tab1
dir="$dataset/$model_name$name"


CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset $dataset\
               --model-name $model_name --train_dir $dir --epochs 1000 --batch_size 256 --activation tanh\
               --scheduler cosine --lr 0.0001 --save_checkpoint_epochs 50 --seed 1\
               --offset 0.0 --temperature 0.25\
               --penalizeCurvature --hessianRegularizerCoefficient 0.1\
               --hessianRegularizerPrimalDualStepSize 0.05 --hessianRegularizerMinimumCoefficient 0.01\
               --hessianRegularizerPrimalDualEpsilon 0.6


CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode certified --dataset $dataset --model-name $model_name\
           --train_dir $dir --activation tanh --plot

# To perform the certification using the optimization method of Sigla & Feizi 2020, use the flag --newtonStep:
#CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode certified --dataset $dataset --model-name $model_name\
#           --train_dir $dir --activation tanh --plot --newtonStep

# PGD attack:
CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode certified_attack --dataset $dataset --model-name $model_name\
            --train_dir $dir --activation $activation

