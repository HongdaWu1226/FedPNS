#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import models.func as fc
# from models.func import node_deleting
import models.test as ts
# from models.test import test_part
import copy
import torch
import operator
import test
from torch import nn
from numpy import linalg as LA
import logging
import torch.nn.functional as F
import numpy as np
logger = logging.getLogger("main_fed")
logger.setLevel(level=logging.DEBUG)
 

def FedAvg(w, idxs_users):
    
    # models = list(w.values())
    w_avg = w[idxs_users[0]]
    for k in w_avg.keys():
        for i in range(1, len(idxs_users)):    
            w_avg[k] += w[idxs_users[i]][k]
        w_avg[k] = torch.div(w_avg[k], len(idxs_users))
    return w_avg

def average(grad_all):

    value_list = list(grad_all.values())
    
    w_avg = copy.deepcopy(value_list[0])
    # print(type(w_avg))
    for i in range(1, len(value_list)):
        w_avg += value_list[i]
    return w_avg / len(value_list)


def Feddel(net_glob, w_locals, gradient, idxs_users, max_now, dataset_test, test_sampler, args, test_count):
    
    full_user = copy.copy(idxs_users)
    # nr_th = len(idxs_users) * 0.7
    gradient.pop('avg_grad')
    expect_list = {}
    labeled = []
    while len(w_locals) > 8:
        expect_list = fc.node_deleting(expect_list, max_now, idxs_users, gradient)
        # print(len(w_locals), expect_list)
        key = max(expect_list.items(), key=operator.itemgetter(1))[0]
        if expect_list[key] <= expect_list["all"]:
            # w_glob = FedAvg(w_locals, idxs_users)
            w_locals, idxs_users
            break
        else:
            labeled.append(key)
            test_count[key][1] += 1
            expect_list.pop("all")
            # print(key)
            loss_all, loss_pop, idxs_users= ts.test_part(net_glob, w_locals, idxs_users, key, dataset_test, test_sampler, args)
            # print(loss_all, loss_pop)
            if loss_all < loss_pop:
                w_locals, idxs_users.append(key)
                break
            else:
                # idxs_users.remove(key)
                w_locals.pop(key)
                gradient.pop(key)
                max_now = expect_list[key]
                expect_list.pop(key)
                # print(idxs_users, len(w_locals), expect_list.keys())

            
                    
#     print(loss_all, loss_pop, worker_ind)
    return w_locals, full_user, idxs_users, labeled, test_count


def Fedbn2(w, gradient):

    g_norm = {}
    for idx in list(gradient.keys()):
        g_norm[idx] = LA.norm(gradient[idx])

    avg_iid, avg_niid = get_avg(g_norm)
    n_items = {k: w[k] for k in list(w)[:10]}

    # logger.info('left %s', sorted(list(n_items.keys())))
    w_agg = FedAvg(w, list(n_items.keys()))
    
    return w_agg, avg_iid, avg_niid

def Diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

def get_avg(g_norm):

    key = list(sorted(g_norm.keys()))
    
    
    iid = []
    for i in range(len(key)):
        if key[i] < 25:
            iid.append(key[i])
    niid = Diff(key, iid)
    # print(iid, niid)

    avg = g_norm[iid[0]]
    for i in range(len(iid)):
        # print('iid', iid[i])
        avg += g_norm[iid[i]]
        
    avg_1 = g_norm[niid[00]]
    for i in range(len(niid)):
        # print('niid', niid[i])
        avg_1 += g_norm[niid[i]]
    return avg/len(iid), avg_1/len(niid)


# below function is for synthetic dataset 

def Feddel_syn(net_glob, w_locals, gradient, idxs_users, max_now, dataset_test, args, test_count):
    
    full_user = copy.copy(idxs_users)
    # nr_th = len(idxs_users) * 0.7
    gradient.pop('avg_grad')
    expect_list = {}
    labeled = []
    while len(w_locals) > 8:
        expect_list = fc.node_deleting(expect_list, max_now, idxs_users, gradient)
        # print(len(w_locals), expect_list)
        key = max(expect_list.items(), key=operator.itemgetter(1))[0]
        if expect_list[key] <= expect_list["all"]:
            # w_glob = FedAvg(w_locals, idxs_users)
            w_locals, idxs_users
            break
        else:
            labeled.append(key)
            test_count[key][1] += 1
            expect_list.pop("all")
            loss_all, loss_pop, idxs_users= test_part(net_glob, w_locals, idxs_users, key, dataset_test, args)
            if loss_all < loss_pop:
                w_locals, idxs_users.append(key)
                break
            else:
                w_locals.pop(key)
                gradient.pop(key)
                max_now = expect_list[key]
                expect_list.pop(key)

    return w_locals, full_user, idxs_users, labeled, test_count

def test_part(net_glob, w_locals, idxs_users, key, dataset_test_part, args):
    net_all = FedAvg(w_locals, idxs_users)
    # loss_all = 1
    net_glob.load_state_dict(net_all)
    acc, loss_all = test(net_glob, dataset_test_part, args)
    idxs_users.remove(key)
    net_part = FedAvg(w_locals, idxs_users)
    net_glob.load_state_dict(net_part)
    acc, loss_part = test(net_glob, dataset_test_part, args)
    # print(loss_all)
    return loss_all, loss_part, idxs_users


def test(net_g, test_data, args):

    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    for i in range(int(args.num_users*args.ratio)):
        data = test_data['user_data'][i]

        for X, y in batch_data(data, args.local_bs):
            
            log_probs = net_g(X)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, y.long(), reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(y.long().data.view_as(y_pred)).long().cpu().sum()

    
    test_loss /= len(test_data['user_data'][0]['y'])*int(args.num_users*args.ratio)
    accuracy = 100 * correct / (len(test_data['user_data'][0]['y'])*int(args.num_users*args.ratio))
  
    return  accuracy, test_loss

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        X = torch.FloatTensor(batched_x)
        y = torch.FloatTensor(batched_y)
        
        yield (X, y)
