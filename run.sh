#!/bin/bash

gpu=0

#
#dataset=mnist
#model_name=liplt-4C3F
#name=lipLt
#dir="$dataset/$model_name$name"
#activation=softplus
#
#CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset $dataset\
#                --model-name $model_name --train_dir $dir --epochs 100 --batch_size 256 --activation $activation\
#                --scheduler cosine --lr 0.0001 --save_checkpoint_epochs 5\
#                --crm --offset 2.3
#                # --penalizeCurvature --hessianRegularizerCoefficient 0.1 --hessianRegularizerPrimalDualStepSize 0.01\
#                # --hessianRegularizerPrimalDualEpsilon 0.5 --hessianRegularizerMinimumCoefficient 0.001 \

dataset=cifar10
model_name=liplt-6F
name=finaltest6Fs2
dir="$dataset/$model_name$name"
activation=tanh

# CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset $dataset\
#                 --model-name $model_name --train_dir $dir --epochs 500 --batch_size 512 --activation $activation\
#                 --scheduler cosine --lr 0.0001 --save_checkpoint_epochs 50 --seed 1\
#                 --offset 0.0 --temperature 1.0 #--crm
                # --penalizeCurvature --hessianRegularizerCoefficient 0.1\
                # --hessianRegularizerPrimalDualStepSize 0.05 --hessianRegularizerMinimumCoefficient 0.0001\
                # --hessianRegularizerPrimalDualEpsilon 0.6


#dataset=cifar100
#model_name=liplt-8C2FCIFAR100
#name=c100E5000
#dir="$dataset/$model_name$name"
#activation=tanh
#
#CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset $dataset\
#               --model-name $model_name --train_dir $dir --epochs 5000 --batch_size 256 --activation $activation\
#               --scheduler cosine --lr 0.001 --save_checkpoint_epochs 10\
#                --offset 0. --temperature 0.25\
#                 --penalizeCurvature --hessianRegularizerCoefficient 0.1 --hessianRegularizerPrimalDualStepSize 0.01\
#                 --hessianRegularizerPrimalDualEpsilon 0.4 --hessianRegularizerMinimumCoefficient 0.001

# CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode certified_attack --dataset $dataset --model-name $model_name\
            #  --train_dir $dir --activation $activation

#CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode attack --dataset $dataset --model-name $model_name\
#             --train_dir $dir --activation $activation
#
#CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode certified --dataset $dataset --model-name $model_name\
#             --train_dir $dir --activation $activation --newtonStep
#
#dataset=cifar10
#model_name=liplt-6C4F
#name=lipLt
#dir="$dataset/$model_name$name"
#activation=centered_softplus
#
#CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset $dataset\
#                --model-name $model_name --train_dir $dir --epochs 100 --batch_size 256 --activation $activation\
#                --scheduler cosine --lr 0.0001 --save_checkpoint_epochs 5\
#                --crm --offset 0.25
#
#dataset=cifar10
#model_name=liplt-9C3F
#name=lipLt
#dir="$dataset/$model_name$name"
#activation=centered_softplus
#
#CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset $dataset\
#                --model-name $model_name --train_dir $dir --epochs 100 --batch_size 256 --activation $activation\
#                --scheduler cosine --lr 0.0001 --save_checkpoint_epochs 5\
#                --crm --offset 0.25
#
#
#dataset=cifar10
#model_name=small
#name=TanhClean
#dir="$dataset/$model_name$name"
#activation=tanh
##
#CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset $dataset\
#               --model-name $model_name --train_dir $dir --epochs 1000 --batch_size 256 --activation $activation\
#               --scheduler cosine --lr 0.0001 --save_checkpoint_epochs 50
            #    --crm --offset 0.25

# CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode certified_attack --dataset $dataset --model-name $model_name\
#              --train_dir $dir --activation $activation

# CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode attack --dataset $dataset --model-name $model_name\
#            --train_dir $dir --activation $activation
#
CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode certified --dataset $dataset --model-name $model_name\
           --train_dir $dir --activation $activation --newtonStep --plot