#!/usr/bin/env python3

from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' 
# ViT_CIFAR10_data when training for CiFAR10
# ViT_CIFAR10_data when training for CiFAR100
DATA_TEST_DIR = DATA_DIR / 'ViT_CIFAR10_data'
# MODEL_DIR = BASE_DIR / 'models'/'models_CiFAR10' for CiFAR10
# MODEL_DIR = BASE_DIR / 'models'/'models_CiFAR100' for CiFAR100
MODEL_DIR = BASE_DIR / 'models' / 'models_CiFAR10'
# DATASET = 'CIFAR10' for CIFAR10
# DATASET = 'CIFAR100' for CIFAR10
DATASET = 'CIFAR10'
# SUBSET = 0 when using the entire dataset
# SUBSET = 1 when using a subset of dataset to train the model on
SUBSET = 0
# NUM_SUBSET_IMAGES = num_of images in the subset per class for training when subset is 1
NUM_SUBSET_IMAGES = 1000
# Number of Images to Fine Tune
FINE_TUNE_SIZE = 1000
config = {
    "patch_size": 4,  
    "hidden_size": 48,
    "num_hidden_layers": 5,
    "num_attention_heads": 12,
    "intermediate_size": 4 * 48, # 4 * hidden_size  (to be also changed when hidden size is changed)
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR100 / CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
# defining dataset/training specific parameters
base_lr = 5e-4
weight_decay = 1e-6
num_classes = 10  # num_classes of CIFAR100 / CIFAR10
accum_iter = 2
tasks = 5
epochs = 40
batch_size = 64
fine_tune_epoch = 5
# hyperparameters specific to BiRT training
alpha_t = 0.005   # controls amount of label noise
alpha_a = 0.005   # controls amount of attention noise
alpha_s = 0.005   # controls amount of trial to trial variability, by applying noise to logits of semantic memory
alpha_e = 0.003   # controls updation of sematic weights
alpha_loss_rep = 0.4  # hyperparameter used in representation loss
rho_loss_cr = 1       # hyperparameter used in calculating total loss
beta_1_loss = 0.05     # hyperparameter used in consistency regulation loss
beta_2_loss = 0.01     # hyperparameter used in consistency regulation loss
_gamma = 0.005          # hyperparameter used for updating semantic memory
percentage_change = 5  # hyperparameter used in label noise
std = 1                 # hyperparameter for normal distribution used in creating noise to be appliedsemntic memory logits
mean = 0                # hyperparameter for normal distribution used in creating noise to be appliedsemntic memory logits
sem_mem_length = 500

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'     # choosong device