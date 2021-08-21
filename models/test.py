#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
import models.Fed as Fed
# import models

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from Fed import average

def test_img(net_g, datatest, args, train_sampler):

    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs, sampler=train_sampler)
    # print(data_loader)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
    
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    
    test_loss /= len(data_loader.dataset)
    
    # accuracy = 100 * correct / len(data_loader.dataset)
    accuracy = correct.item() / 100
    
 
    return accuracy, test_loss

def test_part(net_glob, w_locals, idxs_users, key, dataset_test_part, test_sampler, args):
    net_all = Fed.FedAvg(w_locals, idxs_users)
    # loss_all = 1
    net_glob.load_state_dict(net_all)
    acc, loss_all = test_img(net_glob, dataset_test_part, args, test_sampler)
    idxs_users.remove(key)
    net_part = Fed.FedAvg(w_locals, idxs_users)
    net_glob.load_state_dict(net_part)
    acc, loss_part = test_img(net_glob, dataset_test_part, args, test_sampler)
    # print(loss_all)
    return loss_all, loss_part, idxs_users




