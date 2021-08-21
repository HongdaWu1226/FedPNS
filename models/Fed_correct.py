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

 
# def FedAvg(w, idxs_users):
#     # w_avg = copy.deepcopy(w[0].values())
#     # model_list = list(w.values())
#     # print(w.keys(), idxs_users)
#     # print(idxs_users[0])
#     w_avg = copy.deepcopy(w[idxs_users[0]])
    
#     for i in range(1, len(idxs_users)):
#         # print(idxs_users[i])
#         for k in w_avg.keys():
#         # 
#         # if list(w.keys())[i] in idxs_users:
#             w_avg[k] += w[idxs_users[i]][k]
#         w_avg[k] = torch.div(w_avg[k], len(idxs_users))
#     return w_avg
def FedAvg(w, idxs_users):
    
    # models = list(w.values())
    w_avg = w[idxs_users[0]]
    for k in w_avg.keys():
        for i in range(1, len(idxs_users)):    
            w_avg[k] += w[idxs_users[i]][k]
        w_avg[k] = torch.div(w_avg[k], len(idxs_users))
    return w_avg

def average(grad_all):

    # key = list(grad_all.keys())
    value_list = list(grad_all.values())
    w_avg = copy.deepcopy(value_list[0])
    # print(type(w_avg))
    for i in range(1, len(value_list)):
        w_avg += value_list[i]
    return w_avg / len(value_list)


def Feddel(net_glob, w_locals, gradient, idxs_users, max_now, dataset_test_part, args, test_count):
    
    full_user = copy.copy(idxs_users)
    # nr_th = len(idxs_users) * 0.7
    gradient.pop('avg_grad')
    expect_list = {}
     
    while len(w_locals) > 7:
        expect_list = fc.node_deleting(expect_list, max_now, idxs_users, gradient)
        # print(len(w_locals), expect_list)
        key = max(expect_list.items(), key=operator.itemgetter(1))[0]
        if expect_list[key] <= expect_list["all"]:
            # w_glob = FedAvg(w_locals, idxs_users)
            w_locals, idxs_users
            break
        else:
            test_count[key][1] += 1
            expect_list.pop("all")
            # print(key)
            loss_all, loss_pop, idxs_users= ts.test_part(net_glob, w_locals, idxs_users, key, dataset_test_part, args)
            print(loss_all, loss_pop)
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

            
                # print(len(w_locals), expect_list[key], max_now)
                # if expect_list[key] <= max_now:
                #     # w_glob = FedAvg(w_locals, idxs_users)
                #     w_locals, idxs_users
                # else:
                #     max_now = expect_list[key]
                #     gradient.pop(key)
                #     expect_list.pop(key)
                #     expect_list = fc.node_deleting(expect_list, max_now, idxs_users, gradient)
                #     # print(expect_list)
                #     key = max(expect_list.items(), key=operator.itemgetter(1))[0]
                #     # print(key)
                #     if expect_list[key] <= expect_list["all"]:         
                #         break
                #     else:
                #         loss_all, loss_pop, idxs_users = ts.test_part(net_glob, w_locals, idxs_users, key, dataset_test_part, args)
                #                     # logger.info("loss all '%s' loss pop %s", loss_all, loss_pop)
                #         # print(loss_all, loss_pop)
                #         if loss_all > loss_pop:
                #             w_locals.pop(key)
                #             # idxs_users.pop(key)
                #         break  
                    
#     print(loss_all, loss_pop, worker_ind)
    return w_locals, full_user, idxs_users, test_count