#!/usr/bin/env python3

# importing modules
from src import const
import matplotlib.pyplot as plt
import numpy as np
import einops
import math
from tqdm.notebook import tqdm, trange
from torchsummary import summary
import torch.nn.functional as F
import time
import torch
from torch import nn
import torchvision
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
import torchvision.transforms as transforms


def train_data():
    # define transforms on dataset
    transform_training_data = Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # define path to training dataset
    if const.DATASET == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10(
        root=const.DATA_TEST_DIR, train=True, download=True, transform=transform_training_data)
    else :
        train_data = torchvision.datasets.CIFAR100(
        root=const.DATA_TEST_DIR, train=True, download=True, transform=transform_training_data)
    
  


    # create a dictionary to store the class indices
    class_indices  = {i: [] for i in range(const.num_classes)}

    # seperate indices with respect to classes
    for i in range(int(len(train_data))):
        current_class = train_data[i][1]
        class_indices[current_class].append(i)

    # make datasets for each tasks
    task_list = []

    if (const.SUBSET == 0):
        for index in range(const.tasks):
            class_indices_for_task = sum([class_indices[i] for i in range((index * const.num_classes) // const.tasks, ((index + 1) * const.num_classes) // const.tasks)], [])
            sub_dataset = torch.utils.data.Subset(train_data, class_indices_for_task)
            trainloader = torch.utils.data.DataLoader(sub_dataset, batch_size=const.batch_size, shuffle=True, num_workers=1, drop_last=True)
            task_list.append(trainloader)
    else:
        for index in range(const.tasks):
            class_indices_for_task = sum([class_indices[i][0:const.NUM_SUBSET_IMAGES] for i in range((index * const.num_classes) // const.tasks, ((index + 1) * const.num_classes) // const.tasks)], [])
            sub_dataset = torch.utils.data.Subset(train_data, class_indices_for_task)
            trainloader = torch.utils.data.DataLoader(sub_dataset, batch_size=const.batch_size, shuffle=True, num_workers=1, drop_last=True)
            task_list.append(trainloader)

    return task_list

def test_data():
    # defining transforms on test data
    transform_test_data = Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # defining path to test data
    if const.DATASET == 'CIFAR10':
        test_data = torchvision.datasets.CIFAR10(
        root=const.DATA_TEST_DIR, train=False, download=True, transform=transform_test_data)
    else:
        test_data = torchvision.datasets.CIFAR100(
        root=const.DATA_TEST_DIR, train=False, download=True, transform=transform_test_data)

    # defining dataloader for test data
    ten_percent_test = int(len(test_data))
    subset_test_data = torch.utils.data.Subset(test_data, range(ten_percent_test))
    testloader = torch.utils.data.DataLoader(subset_test_data, batch_size=const.batch_size,
                                            shuffle=False, num_workers=1, drop_last = True)
    return testloader

def balanced_fine_tune_data():
    # define transforms on dataset
    transform_training_data = Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # define path to training dataset
    if const.DATASET == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10(
        root=const.DATA_TEST_DIR, train=True, download=True, transform=transform_training_data)
    else :
        train_data = torchvision.datasets.CIFAR100(
        root=const.DATA_TEST_DIR, train=True, download=True, transform=transform_training_data)
    
    indices = torch.randperm(len(train_data))[:const.FINE_TUNE_SIZE]
    subset_train_data = torch.utils.data.Subset(train_data, indices)
    fine_tune_dataloader = torch.utils.data.DataLoader(subset_train_data, batch_size=const.batch_size,
                                            shuffle=True, num_workers=1, drop_last = True)
    
    return fine_tune_dataloader