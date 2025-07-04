# learning-to-defer-to-a-population-with-limited-demonstrations
Implementation of Learning to Defer to a population with limited expert demonstration


python train_embedding_fm.py --exp-dir expert_0  --n-labeled 500 --ex_strength 2  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 500 --ex_strength 5  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 5 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 500 --ex_strength 8  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 8 --with-attn attn --finetune &&

python train_embedding_fm.py --exp-dir expert_0  --n-labeled 500 --ex_strength 2  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 500 --ex_strength 5  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 5 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 500 --ex_strength 8  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 8 --with-attn attn --finetune &&

python train_embedding_fm.py --exp-dir expert_0  --n-labeled 473 --ex_strength 8  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 8 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 473 --ex_strength 21  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 21 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 473 --ex_strength 30  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 30 --with-attn attn --finetune &&

python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 1  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 1 --with-attn mlp --finetune &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 5  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 5 --with-attn mlp --finetune &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 9  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 9 --with-attn mlp --finetune &&

python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 1  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 1 --with-attn mlp --finetune &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 5  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 5 --with-attn mlp --finetune &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 500 --ex_strength 9  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 9 --with-attn mlp --finetune &&

python train_embedding_fm.py --exp-dir expert_1  --n-labeled 473 --ex_strength 4  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 4 --with-attn mlp --finetune &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 473 --ex_strength 21  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 21 --with-attn mlp --finetune &&
python train_embedding_fm.py --exp-dir expert_1  --n-labeled 473 --ex_strength 38  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 38 --with-attn mlp --finetune


<<<<<<< HEAD
bash train_gtsrb.sh single 21 train 0 473 w H && bash train_gtsrb.sh single 30 train 0 473 w H
bash fashion.sh single 2 train 0 500 w H && bash fashion.sh single 5 train 0 500 w H && bash fashion.sh single 8 train 0 500 w H
bash train_cifar10.sh single 2 train 0 500 w H && bash train_cifar10.sh single 5 train 0 500 w H && bash train_cifar10.sh single 8 train 0 500 w H
=======

bash train_generated_experts_gtsrb.sh single 21 train 0 473 w H && bash train_generated_experts_gtsrb.sh single 30 train 0 473 w H
bash train_generated_experts_fashion.sh single 2 train 0 500 w H && bash train_generated_experts_fashion.sh single 5 train 0 500 w H && bash train_generated_experts_fashion.sh single 8 train 0 500 w H
bash train_generated_experts_cifar.sh single 2 train 0 500 w H && bash train_generated_experts_cifar.sh single 5 train 0 500 w H && bash train_generated_experts_cifar.sh single 8 train 0 500 w H
<<<<<<< HEAD
>>>>>>> 318f51fafba8e9b2a3b03a5f57f842c54fb65e1d
=======


Accuracy v/s L experiments:

CIFAR 
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 20 --ex_strength 2  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune && 
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 40 --ex_strength 2  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 60 --ex_strength 2  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 100 --ex_strength 2  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 200 --ex_strength 2  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 500 --ex_strength 2  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 2500 --ex_strength 2  --dataset CIFAR10 --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune

FASHION
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 20 --ex_strength 2  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune && 
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 40 --ex_strength 2  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 60 --ex_strength 2  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 100 --ex_strength 2  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 200 --ex_strength 2  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 500 --ex_strength 2  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 2500 --ex_strength 2  --dataset FASHION --n-epoches 50 --batchsize 64 --seed 0  --p-out 2 --with-attn attn --finetune

GTSRB
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 86 --ex_strength 8  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 8 --with-attn attn --finetune && 
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 172 --ex_strength 8  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 8 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 258 --ex_strength 8  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 8 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 430 --ex_strength 8  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 8 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled  860 --ex_strength 8  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 8 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 500 --ex_strength 8  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 8 --with-attn attn --finetune &&
python train_embedding_fm.py --exp-dir expert_0  --n-labeled 2150 --ex_strength 8  --dataset GTSRB --n-epoches 50 --batchsize 64 --seed 0  --p-out 8 --with-attn attn --finetune

>>>>>>> cb162ad6151aad1fd517dc704d133968bd21b203
