#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def Dataset_config(dataset, num_users, ratio, num_sample, pattern):

    if dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST(root = './data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST(root='./data/mnist/', train=False, download=True, transform=trans_mnist)
 
    elif dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

    else: 
        exit('Error: unrecognized dataset')
            
    if pattern == 'iid':
        dict_users = iid(dataset_train, num_users, 1, num_sample)

    else:
        dict_users = iid(dataset_train, num_users, ratio, num_sample)
        non_iid = noniid(dataset_train, num_users, ratio, num_sample, dataset)
        dict_users.update(non_iid)

    all_idxs = [i for i in range(len(dataset_train))]
    all_idxs_test = [i for i in range(len(dataset_test))]
    train_sampler = SubsetRandomSampler(all_idxs[:5000])
    test_sampler = SubsetRandomSampler(all_idxs_test)
    test_sampler_temp = SubsetRandomSampler(all_idxs_test[:2000])


    
    # print(dict_users)
    return dict_users, dataset_train, dataset_test, train_sampler, test_sampler, test_sampler_temp
    # , dataset_test_part


def iid(dataset, num_users, ratio, num_sample):
    """
    Sample I.I.D. client data from MNIST dataset
    """
    # num_items = int(len(dataset)/num_users)
    # num_items = 10
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # print(all_idxs)
    for i in range(int(num_users*ratio)):
        
        dict_users[i] = set(np.random.choice(all_idxs, num_sample, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    # print(dict_users[0])
    return dict_users


def noniid(dataset, num_users, ratio, num_sample, name):
    """
    Sample non-I.I.D client data from MNIST dataset
    :return:
    """
    #if two shard concatenate
    # num_imgs = int(num_sample/2)   
    num_imgs = int(num_sample)
    num_shards = int(len(dataset)/ num_imgs)
    # num_shards, num_imgs = 600, int(num_sample/2)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(int(num_users*ratio),num_users)}
    idxs = np.arange(num_shards*num_imgs)

    if name == 'mnist':
        labels = dataset.train_labels.numpy()
    else:
        labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(int(num_users*ratio), num_users):
        
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))  # 2  if two label
        # print(rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            dict_users[i] = idxs[rand*num_imgs:(rand+1)*num_imgs]

    return dict_users


