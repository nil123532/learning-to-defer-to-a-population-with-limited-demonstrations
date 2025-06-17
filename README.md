# learning-to-defer-to-a-population-with-limited-demonstrations
Implementation of Learning to Defer to a population with limited expert demonstration


python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 1  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 1 --with-attn attn &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 5  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 5 --with-attn attn &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 9  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 9 --with-attn attn &&

python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 1  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 1 --with-attn attn &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 5  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 5 --with-attn attn &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 9  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 9 --with-attn attn &&

python train_embedding_fm.py --exp-dir expert_1  --n-labeled 473 --ex_strength 4  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 4 --with-attn attn &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 473 --ex_strength 21  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 21 --with-attn attn &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 473 --ex_strength 38  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 38 --with-attn attn &&

python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 1  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 1 --with-attn mlp &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 5  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 5 --with-attn mlp &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 9  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 9 --with-attn mlp &&

python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 1  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 1 --with-attn mlp &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 5  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 5 --with-attn mlp &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 9  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 9 --with-attn mlp &&

python train_embedding_fm.py --exp-dir expert_1  --n-labeled 473 --ex_strength 4  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 4 --with-attn mlp &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 473 --ex_strength 21  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 21 --with-attn mlp &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 473 --ex_strength 38  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 38 --with-attn mlp