# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


class DoubleCIFAR10(torchvision.datasets.CIFAR10):
    """
    Class that returns two copies of a CIFAR image, but is zeroed out from sides to create two separate views.
    Args:
        view_size: specfies the remaining width of the image after zeroing out remaining pixels.
        transform: transform applied to one view
        transform_b: trasnform applied to other view
        is_rand_zero_input: Zeroes out input from one modality for small fraction of examples (
    """

    def __init__(self, root, train=True,
                 transform=None, transform_b=None,
                 target_transform=None, download=False,
                 is_rand_zero_input=False,
                 view_size=14):
        super(DoubleCIFAR10, self).__init__(
            root,
            train=train,
            transform=None, target_transform=target_transform,
            download=download)
        self.transform = transform
        self.transform_b = transform_b
        self.is_rand_zero_input = is_rand_zero_input
        self.view_to_zero = 32 - view_size

        self.default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.transform_b is not None:
            img_b = self.transform_b(img)

        img = self.default_transform(img)
        img_b = self.default_transform(img_b)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Maybe change the dimensions here and re-run?
        img[:, :, -self.view_to_zero:] = 0.  # used to only be 14
        img_b[:, :, :self.view_to_zero] = 0.

        # Want to zero out to have in data distribution to evaluate information in each view
        if self.train and self.is_rand_zero_input:
            rand_num = np.random.rand(1)
            if rand_num > 0.9:
                img[:, :, :] = 0.
            elif rand_num < 0.1:
                img_b[:, :, :] = 0.

        return img, img_b, target


class DoubleCIFAR10Independent(DoubleCIFAR10):
    """
    Class that returns two copies of a CIFAR image, but is zeroed out from sides to create two separate views.
    Inherits DoubleCIFAR10, but also applies the Flip Augmentation to both views
    """

    def __getitem__(self, index):
        # Call this with augment
        flip = np.random.rand(1) < 0.5  # Stores whether to flip all images or not

        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.transform_b is not None:
            img_b = self.transform_b(img)

        if flip:  # Need to apply to both, apply before normalization from default transform
            img = TF.hflip(img)
            img_b = TF.hflip(img_b)

        img = self.default_transform(img)
        img_b = self.default_transform(img_b)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Maybe change the dimensions here and re-run?
        img[:, :, -self.view_to_zero:] = 0.  # used to only be 14
        img_b[:, :, :self.view_to_zero] = 0.

        # Want to zero out to have in data distribution to evaluate information in each view
        if self.train and self.is_rand_zero_input:
            rand_num = np.random.rand(1)
            if rand_num > 0.9:
                img[:, :, :] = 0.
            elif rand_num < 0.1:
                img_b[:, :, :] = 0.

        return img, img_b, target


def get_cifar_loaders_s(workers=0, batch_size=128, augment=True, deficit=[], val_batch_size=100,
                        is_rand_zero_input=False, view_size=14):
    """
    Get the dataloaders for the blurred images experiment
    """
    transform = []
    if 'downsample' in deficit:
        transform += [transforms.Resize(8, antialias=True), transforms.Resize(32)]
    transform = transforms.Compose(transform)

    transform_train = []
    if augment and 'noise' not in deficit:
        transform_train += [
            transforms.Pad(padding=4, fill=(125, 123, 113)),
            transforms.RandomCrop(32, padding=0),
            transforms.RandomHorizontalFlip()]
    transform_train = transforms.Compose(transform_train)

    transform_test = []
    transform_test = transforms.Compose(transform_test)

    # Note only 1 view is being augmented right now!
    trainset = DoubleCIFAR10(root=os.path.join(os.environ['HOME'], 'data'), train=True, download=False,
                             transform=transform_train, transform_b=transform, is_rand_zero_input=is_rand_zero_input,
                             view_size=view_size)
    testset = DoubleCIFAR10(root=os.path.join(os.environ['HOME'], 'data'), train=False, download=False,
                            transform=transform_test, transform_b=transform, is_rand_zero_input=is_rand_zero_input,
                            view_size=view_size)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return trainloader, testloader


def get_cifar_loaders_ind(workers=0, batch_size=128, augment=True, deficit=[], val_batch_size=100,
                          is_rand_zero_input=False, view_size=14, is_diff_aug=False):
    """
    Allow independent Augmentation on each view
    With RandomHorizontalflip, and flip both views theres a chance that the views show same thing.
    Args:
        is_diff_aug: True if use separate augmentation for each view
    """
    transform = []
    transform = transforms.Compose(transform)

    transform_train = []
    if augment and 'noise' not in deficit:
        transform_train += [
            transforms.Pad(padding=4, fill=(125, 123, 113)),
            transforms.RandomCrop(32, padding=0),
            # transforms.RandomHorizontalFlip()
        ]
    transform_train = transforms.Compose(transform_train)

    transform_test = []
    transform_test = transforms.Compose(transform_test)

    trainset = DoubleCIFAR10Independent(root=os.path.join(os.environ['HOME'], 'data'), train=True, download=False,
                                        transform=transform_train,
                                        transform_b=transform if is_diff_aug else transform_train,
                                        is_rand_zero_input=is_rand_zero_input, view_size=view_size)
    testset = DoubleCIFAR10Independent(root=os.path.join(os.environ['HOME'], 'data'), train=False, download=False,
                                       transform=transform_test, transform_b=transform,
                                       is_rand_zero_input=is_rand_zero_input,
                                       view_size=view_size)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return trainloader, testloader
