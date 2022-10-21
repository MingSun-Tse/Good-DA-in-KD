from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import torch
import copy
import math
"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""
def get_data_from_npy(npy_set):
    if npy_set.endswith('.npy'):
        npyfile = npy_set
    else:
        npyfile = os.path.join(npy_set, 'batch.npy')
    data = np.load(npyfile, allow_pickle=True)
    images = [np.array(x) for x in data[:, 0]]
    images = np.array(images)
    labels = np.array(data[:, 1])
    return images, labels

def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

NEW_TORCH = int(torch.__version__.split('.')[0]) > 0
class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        if NEW_TORCH: # see https://github.com/HobbitLong/RepDistiller/issues/4
            img, target = self.data[index], self.targets[index]
        else:
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar100_dataloaders(batch_size=128, num_workers=8, is_instance=False, npy_set='', augment=False, use_DA='11', opt=None):
    """
    cifar 100
    """
    data_folder = get_data_folder()
    transforms_ = []
    if use_DA[0] == '1':
        transforms_ += [transforms.RandomCrop(32, padding=4)]
    if use_DA[1] == '1':
        transforms_ += [transforms.RandomHorizontalFlip()]
    transforms_ += [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
    ]
    train_transform = transforms.Compose(transforms_)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    if is_instance:
        train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
    else:
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    
    train_loader2 = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, train_loader2, test_loader, n_data
    else:
        return train_loader, train_loader2, test_loader

class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        self.num_classes = 100
        
        # -- @mst: maintain the same interface due to PyTorch version change
        # see https://github.com/HobbitLong/RepDistiller/issues/4
        if NEW_TORCH:
            if self.train:
                self.train_data = self.data
                self.train_labels = self.targets
            else:
                self.test_data = self.data
                self.test_labels = self.targets
        # --

        self.n_data = len(self.train_data) if self.train else len(self.test_data)
        if self.train:
            num_samples = len(self.train_data)
            label = self.train_labels
        else:
            num_samples = len(self.test_data)
            label = self.test_labels
        self.label = np.array(label)

        self.cls_positive = [[] for i in range(self.num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(self.num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(self.num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(self.num_classes)]

        self.cls_positive = np.asarray(self.cls_positive) # shape: 100 x 500
        self.cls_negative = np.asarray(self.cls_negative) # shape: 100 x 49500 for cifar100 (every class has 500 images)
    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx)) # insert pos_idx to the top of neg_idx
            return img, target, index, sample_idx

MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)
def get_cifar100_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0,
                                    use_DA='11', opt=None):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    transforms_ = []
    if use_DA[0] == '1':
        transforms_ += [transforms.RandomCrop(32, padding=4)]
    if use_DA[1] == '1':
        transforms_ += [transforms.RandomHorizontalFlip()]
    transforms_ += [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
    ]
    train_transform = transforms.Compose(transforms_)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = CIFAR100InstanceSample(root=data_folder,
                                    download=True,
                                    train=True,
                                    transform=train_transform,
                                    k=k,
                                    mode=mode,
                                    is_sample=is_sample,
                                    percent=percent)
    
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data, train_set
