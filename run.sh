#!/bin/bash

gpu=0

dataset=mnist
model_name=xsmall
dir="$dataset/$model_name"
activation=tanh

CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset $dataset \
             --model-name $model_name --train_dir $dir --epochs 250 --batch_size 256 --activation $activation \
             --penalizeCurvature --hessianRegularizerPrimalDualEpsilon 0.75



CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode certified --dataset $dataset --model-name $model_name\
             --train_dir $dir --activation $activation