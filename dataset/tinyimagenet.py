from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
from torchvision import datasets, transforms
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# refer to: https://github.com/pytorch/examples/blob/master/imagenet/main.py
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=8), # refer to the cifar case
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

class TinyImageNet(data.Dataset):
    def __init__(self,     root, transform):
        self.data = np.load(os.path.join(root, "batch.npy"), allow_pickle=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index][0])
        img = self.transform(img)
        label = self.data[index][1]
        label = torch.LongTensor([label])[0]
        return img.squeeze(0), label

    def __len__(self):
        return len(self.data) 

class TinyImageNetInstance(data.Dataset):
    """TinyImageNet dataset, instance version
    """
    def __init__(self, root, transform):
        self.data = np.load(os.path.join(root, "batch.npy"), allow_pickle=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index][0])
        img = self.transform(img)
        label = self.data[index][1]
        label = torch.LongTensor([label])[0]
        return img.squeeze(0), label, index

    def __len__(self):
        return len(self.data) 

class TinyImageNetInstanceSample(data.Dataset):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        # --------- load tinyimagenet data in npy format
        folder = 'train' if train else 'test'
        npy = np.load(os.path.join(root, folder, "batch.npy"), allow_pickle=True)
        self.data, self.labels = [], []
        for x in npy:
            self.data.append(x[0])
            self.labels.append(x[1])
        self.data = np.asarray(self.data)
        self.labels = np.asarray(self.labels)
        self.labels = torch.from_numpy(self.labels).long() 
        print('data shape:', self.data.shape)
        print('label shape:', self.labels.shape)
        # ---------
        self.transform = transform
        self.target_transform = target_transform

        num_classes = 200
        num_samples = len(self.data)

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[self.labels[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = np.reshape(img, [32,32,3])
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
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
    
    def __len__(self):
        return len(self.data)


def get_tinyimagenet_dataloaders(batch_size=64, num_workers=8, is_instance=False):
    data_folder = './data/tinyimagenet'
    # train set and loder
    if is_instance:
        train_set = TinyImageNetInstance(root=data_folder+'/train', transform=transform_train)
        n_data = len(train_set)
    else:
        train_set = TinyImageNet(root=data_folder+'/train', transform=transform_train)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    # test set and loder
    test_set = TinyImageNet(root=data_folder+'/val', transform=transform_test)
    test_loader = DataLoader(test_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


def get_tinyimagenet_dataloaders_sample(batch_size=64, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    data_folder = './data/tinyimagenet'
    train_set = TinyImageNetInstanceSample(root=data_folder,
                                       train='True',
                                       transform=transform_train,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
                              
    test_set = TinyImageNet(root=data_folder+'/val', transform=transform_test)
    test_loader = DataLoader(test_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    return train_loader, test_loader, n_data, train_set
