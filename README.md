# Running the Experiments

This guide explains how to set up the environment, generate pseudo-labels, and run **Learning to Defer to a Population (L2D-POP)** experiments.

---

## 1. Environment Setup

First, create a dedicated conda environment with Python 3.9:

```bash
conda create -n limited_pop_l2d python=3.9
conda activate limited_pop_l2d
```

Then, install the required dependencies:

```bash
# Install PyTorch
conda install pytorch torchvision torchaudio -c pytorch -c nvidia

# Install core libraries
conda install numpy scipy matplotlib jupyterlab jupyter_console jupyter_client scikit-learn

# Additional packages
pip install attrdict tensorboard_logger timm tensorboard pandas opencv-python seaborn
```

---

## 2. Generate Pseudo-Labels

All scripts for generating pseudo-labels are located in the `generating-pseudo-labels` directory.

### Step 2.1: Pre-train the Embedding Model

Run the following command to pre-train the embedding model on your chosen dataset:

```bash
python train_emb_model.py --dataset CHOSEN_DATASET --model wideresnet --num_classes NUM_CLASSES --batch 128 --lr 0.001
```

- `CHOSEN_DATASET ∈ {cifar10, gtsrb, fashion}`
- `num_classes`: number of classes in the dataset

---

### Step 2.2: Generate Labels

Once pre-training is complete, generate labels using:

```bash
python train_embedding_fm.py --exp-dir expert_0 --n-labeled N_LABELED --ex_strength EX_STRENGTH \
    --dataset CHOSEN_DATASET --n-epoches 50 --batchsize 64 --seed 0 --p-out P_OUT --with-attn attn
```

**Arguments:**
- `N_LABELED`: number of labeled samples per expert (range: 20–2500 depending on dataset)  
- `CHOSEN_DATASET ∈ {CIFAR10, GTSRB, FASHION}`
- `P_OUT`: {8, 34}, depending on the dataset  
- `EX_STRENGTH`: {8, 34}, depending on the dataset  

Repeat for all desired `N_LABELED` values for your chosen dataset.

⚠️ **Note:** Sometimes the embedding model checkpoint may be located in a different directory than expected.  
If this happens, simply copy-paste the checkpoint from the pre-training step (Step 2.1) into the required directory before running this command.

---

### Step 2.3: Create `.npy` Files

After generating labels for the full population:

- Use the appropriate script in the `artificial_expert_labels` directory to create `.npy` files:
  - `cifar.py`
  - `fashion.py`
  - `gtsrb.py`

- Move the generated `.npy` files into the `l2d-pop` directory using the corresponding `move_*.py` script in `artificial_expert_labels` directory.
- Then use the `move_*.py` script in the main directory to move pre-trained checkpoints for the context embedder.

---

## 3. Training L2D-POP

Once setup is complete, switch to the `l2d-pop` directory.

### Train with Ground Truth Labels

```bash
python train_CHOSEN_DATASET.py L2D_METHOD P_OUT MODE 0 ANY_NUM n H
```

### Train with Generated Labels

```bash
python train_generated_experts_CHOSEN_DATASET.py L2D_METHOD P_OUT MODE 0 N_LABELED n H
```

**Arguments:**
- `CHOSEN_DATASET`: cifar10, gtsrb, or fashion  
- `L2D_METHOD`: specify the learning-to-defer method either {pop_attn,pop,single}
- `P_OUT`: {8, 34}, depending on the dataset  
- `MODE`: {train,test}
- `ANY_NUM / N_LABELED`: number of labeled samples used in training  
- `H`: specifies the expert type
