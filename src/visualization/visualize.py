#!/usr/bin/env python3

from src import const
from src.dataset import train_data
import matplotlib.pyplot as plt
import torchvision
import numpy as np

# functions to visualize an image


def imshow2(img):
  img = img / 5 + 0.5     # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    task_list = train_data()
    # plotting 2 batches of the first task
    dataiter = iter(task_list[0])
    images1, labels1 = next(dataiter)
    images2,labels2 = next(dataiter)
    print(images1.size())
    print(labels1.size())

    imshow(torchvision.utils.make_grid(images1))
    print(' '.join(f'{[labels1[j]]:5s}' for j in range(const.batch_size)))
    imshow(torchvision.utils.make_grid(images2))
    print(' '.join(f'{[labels2[j]]:5s}' for j in range(const.batch_size)))