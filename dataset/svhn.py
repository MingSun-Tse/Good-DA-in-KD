from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

def get_data_folder():
    """
    return server-dependent path to store the data
    """
    data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)
class SVHNInstance(datasets.SVHN):
	"""SVHNInstance Dataset.
    """
	def __getitem__(self, index):
	    if self.split == 'train':
	        img, target = self.data[index], self.labels[index]
	    else:
	        img, target = self.data[index], self.labels[index]

	    # doing this so that it is consistent with all other datasets
	    # to return a PIL Image

	    img = np.reshape(img, [32,32,3])

	    # ipdb.set_trace()
	    # print(img.shape)
	    img = Image.fromarray(img)

	    if self.transform is not None:
	        img = self.transform(img)

	    if self.target_transform is not None:
	        target = self.target_transform(target)

	    return img, target, index

def get_svhn_dataloaders(batch_size=200, num_workers=8, is_instance=False):
	"""
    SVHN
    """
	data_folder = get_data_folder()

	train_transform = transforms.Compose([
	    transforms.RandomCrop(32, padding=4),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize(MEAN, STD),
	])
	test_transform = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize(MEAN, STD),
	])

	if is_instance:
	    train_set = SVHNInstance(root=data_folder,
	                                 download=True,
	                                 split='train',
	                                 transform=train_transform)
	    n_data = len(train_set)
	else:
	    train_set = datasets.SVHN(root=data_folder,
	                                  download=True,
	                                  split='train',
	                                  transform=train_transform)
	train_loader = DataLoader(train_set,
	                          batch_size=batch_size,
	                          shuffle=True,
	                          num_workers=num_workers)

	test_set = datasets.SVHN(root=data_folder,
	                             download=True,
	                             split='test',
	                             transform=test_transform)
	test_loader = DataLoader(test_set,
	                         batch_size=int(batch_size/2),
	                         shuffle=False,
	                         num_workers=int(num_workers/2))

	if is_instance:
	    return train_loader, test_loader, n_data
	else:
	    return train_loader, test_loader


class SVHNInstanceSample(datasets.SVHN):
    """
    SVHNInstance+Sample Dataset
    """
    def __init__(self, root, train=True,
	             transform=None, target_transform=None,
	             download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
	    super().__init__(root=root, split='train', download=download,
	                     transform=transform, target_transform=target_transform)
	    self.k = k
	    self.mode = mode
	    self.is_sample = is_sample

	    num_classes = 10
	    if self.split == 'train':
	        num_samples = len(self.data)
	        label = self.labels
	    else:
	        num_samples = len(self.data)
	        label = self.labels

	    self.cls_positive = [[] for i in range(num_classes)]
	    for i in range(num_samples):
	        self.cls_positive[label[i]].append(i)

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
	    if self.split == 'train':
	        img, target = self.data[index], self.labels[index]
	    else:
	        img, target = self.data[index], self.labels[index]

	    # doing this so that it is consistent with all other datasets
	    # to return a PIL Image
	    img = np.reshape(img, [32,32,3])
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

def get_svhn_dataloaders_sample(batch_size=200, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    svhn
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(MEAN, STD),
	])

    train_set = SVHNInstanceSample(root=data_folder,
	                                   download=True,
	                                   train='True',
	                                   transform=train_transform,
	                                   k=k,
	                                   mode=mode,
	                                   is_sample=is_sample,
	                                   percent=percent)
    n_data = len(train_set)

	# kd-Huan: use the test set below instead of this one
    # test_set = SVHNInstance(root=data_folder,
    # 								download=True,
    # 								split='test',
    # 								transform=test_transform)
    test_set = datasets.SVHN(root=data_folder,
	                             download=True,
	                             split='test',
	                             transform=test_transform)

    train_loader = DataLoader(train_set,
	                          batch_size=batch_size,
	                          shuffle=True,
	                          num_workers=num_workers)

    test_loader = DataLoader(test_set,
	                         batch_size=int(batch_size/2),
	                         shuffle=False,
	                         num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data, train_set
