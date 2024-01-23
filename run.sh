gpu=0


# CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --dataset cifar10\
#             --model-name xsmall --train_dir c10/xsmall --epochs 500 --batch_size 256 --activation tanh\
#             --penalizeCurvature True



CUDA_VISIBLE_DEVICES=$gpu python3 main.py --mode certified --dataset cifar10 --model-name xsmall --train_dir c10/xsmall --activation tanh