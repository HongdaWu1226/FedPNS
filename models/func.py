# from  import average
import models.Fed as Fed
    # models.Fed import average
import logging
import torch
import math
import numpy as np
from numpy.random import choice
import torch.nn.functional as F
import pickle
import random
import heapq
from numpy import linalg as LA
from collections import Counter

logger = logging.getLogger("main_fed")
logger.setLevel(level=logging.DEBUG)

def result_avg(result):

    result_list = list(result.values())

    sums = Counter()
    counters = Counter()
    for itemset in result_list:
        sums.update(itemset)
        counters.update(itemset.keys())

    result = {x: float(sums[x])/counters[x] for x in sums.keys()}

    return result



 

def subset(letter_ind, n):
    name_alphabet = list(map(chr, range(letter_ind, letter_ind+n)))
    return name_alphabet

## save variable
def save_obj(obj1, obj2, name):  
    pkl_path = "./review_result/niid/test/"
    with open(pkl_path + name + ".pkl", 'wb') as f:
        pickle.dump([obj1, obj2], f, pickle.HIGHEST_PROTOCOL)

def save_obj_more(obj1, obj2, obj3, name):  
    pkl_path = "./review_result/niid/test/"
    with open(pkl_path + name + ".pkl", 'wb') as f:
        pickle.dump([obj1, obj2, obj3], f, pickle.HIGHEST_PROTOCOL)

# def save_obj_more(obj1, obj2, obj3,obj4,obj5, name):  
#     pkl_path = "./review_result/iid/"
#     with open(pkl_path + name + ".pkl", 'wb') as f:
#         pickle.dump([obj1, obj2, obj3, obj4, obj5], f, pickle.HIGHEST_PROTOCOL)

def save_act_node(obj1, name):  
    pkl_path = "./review/"
    with open(pkl_path + name + ".pkl", 'wb') as f:
        pickle.dump([obj1], f, pickle.HIGHEST_PROTOCOL)

## load variable
def load_obj(name):
    pkl_path = "./result/test/"
    with open(pkl_path + name + ".pkl", 'rb') as f:
        return pickle.load(f)



def model_ini(traced_model):
    initial_model = {}
    params = traced_model.named_parameters()
    for name1, param1 in params:

        initial_model[name1] = param1.data

    return initial_model


def dot(K,L):
    temp = [i[0] * i[1] for i in zip(K, L)]
    ratio = temp.count(1) / len(temp)

    return round(ratio,2)
    
def dot_sum(K, L):

    return round(sum(i[0] * i[1] for i in zip(K, L)),2)



def node_deleting(expect_list, expect_value, worker_ind, grads):

    # expect_list.pop("all")
    for i in range(len(worker_ind)):

        worker_ind_del  = [n for n in worker_ind if n != worker_ind[i]]    
        grad_del = grads.copy()
        grad_del.pop(worker_ind[i])
        avg_grad_del = Fed.average(grad_del)
        grad_del['avg_grad'] = avg_grad_del
        expect_value_del = get_relation(grad_del, worker_ind_del)
        expect_list[worker_ind[i]] = expect_value_del
    expect_list['all'] =  expect_value

    return expect_list

  
def Diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))


def get_gradient(args, pre, now, lr):
 
    grad = np.subtract( model_convert(pre), model_convert(now)) 
   
    return grad / (args.num_sample * args.local_ep * lr / args.local_bs)

def get_relation(avg_grad, idxs_users):

    innnr_value = {}
    for i in range(len(idxs_users)):
        
        innnr_value[idxs_users[i]] = dot_sum(avg_grad[idxs_users[i]], avg_grad['avg_grad'])

    return round(sum(list(innnr_value.values())), 3)

def model_convert(model_grad):

    ini = []
    for name in model_grad.keys():
     
        ini = ini + torch.flatten(model_grad[name]).tolist() 

    return ini


def probabilistic_selection(node_prob, node_count, act_indx, part_node_after, labeled, alpha):
    logger = logging.getLogger("main_fed")
    logger.setLevel(level=logging.DEBUG)

    remove_list = Diff(act_indx, part_node_after)
    
    for i in remove_list:
        node_count[i][2] += 1

    # rest_nodes = Diff(list(node_prob.keys()), remove_list)

    rest_nodes = Diff(list(node_prob.keys()), labeled)
    beta = 0.7
    weight = 0
 
        
    ratio = {}
    for i in labeled:
        ratio[i] = node_count[i][1]/ node_count[i][0]
        
    for i in labeled:
        prob_change =  node_prob[i] * min( (ratio[i] + beta)**alpha, 1)
        logger.info(" node %s, rate_%s, change dis %s", i, node_count[i][1]/ node_count[i][0], min( (ratio[i] + beta)**alpha, 1) )
        weight += prob_change
        node_prob[i] =  node_prob[i] - prob_change
 
    for i in rest_nodes:
            node_prob[i] = node_prob[i] + weight / (len(rest_nodes))


    get_node = choice(list(node_prob.keys()), 10, replace=False, p=list(node_prob.values()))

 
    # print(node_explor)
    return get_node.tolist(), node_prob, node_count


def get_norm(graident):

    vari_norm = {}
    norm_vari = {}
    
    node_indx = list(graident.keys())
    node_indx.remove('avg_grad')

    for idx in list(graident.keys()):
        vari_norm[idx] = LA.norm(graident[idx])
        
    vari_norm = (sum(vari_norm.values()) - vari_norm['avg_grad']) / (len(vari_norm)-1) / vari_norm['avg_grad']

    for idx in node_indx:
        norm_vari[idx] = LA.norm( (graident[idx] - graident['avg_grad']) )
        
   

    return vari_norm, sum(norm_vari.values())/len(norm_vari)