#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
####

import matplotlib
import logging
import matplotlib.pyplot as plt
import copy
import numpy as np
import random
from torchvision import datasets, transforms
import torch
from utils.sampling import Dataset_config
from utils.options import args_parser
from models.func import get_gradient
from models.func import get_relation
from models.func import probabilistic_selection
from models.func import save_obj
from models.func import save_obj_more
from models.func import load_obj
from models.func import Diff
from models.func import get_norm
from models.Update import LocalUpdate
from models.Nets import Net_config
from models.Fed import FedAvg
from models.Fed import average
from models.Fed import Feddel
from models.Fed import Fedbn2
from models.test import test_img
from tqdm import tqdm

if __name__ == '__main__':

    logger = logging.getLogger("main_fed")
    logger.setLevel(level=logging.DEBUG)
    args = args_parser()
    logging.basicConfig(filename = "./result/fig3_mlr_cnnM/" + "%s_%s_%s_%s_%s_%s"
                        %(args.algorithm, args.dataset, args.model, args.local_ep, args.ratio, args.frac) + ".txt")
    # parse args
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    dict_users, dataset_train, dataset_test, train_sampler, test_sampler, test_sampler_temp= Dataset_config(args.dataset, args.num_users, args.ratio, args.num_sample, args.pattern)
    
    # print(len(dataset_train))
    img_size = dataset_train[0][0].shape

    
    loss_all = {}
    acc_all = {}
    for exp in tqdm(range(1, args.num_exp+1)):
        logger.info("--------------Experiment-------------- %s/%s", exp, args.num_exp)
    # build model
        net_glob = Net_config(args, args.model, args.dataset, args.num_classes, args.device, img_size)
        logger.info("my model %s", net_glob)
        net_glob.train()

        # copy weights
        w_glob = net_glob.state_dict()

        # training
        loss_train = []
        acc_test = []
        loss_test = []
        norm_iid = []
        norm_niid = []
        vari_norm_round = []
        norm_vari_round = []
        if args.all_clients: 
            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(args.num_users)]
        learning_rate = args.lr

        node_prob = {}
        # node_count = {}
        test_count = {}
        # all_user = list(range(args.num_users))
            # participate_count = {}
        for i in range(args.num_users):
            node_prob[i] = 1 / args.num_users
            # node_count[i] = 0 
            tupe = []
            for j in range(3):
                tupe.append(0)
                test_count[i] = tupe
        whichnode = {}
        remove_who = {}
        for iter in range(1, args.rounds+1):
            
            loss_locals = []
            if not args.all_clients:
                w_locals = {}
                gradient = {}
            # m = max(int(args.frac * args.num_users), 1)
            
            if args.algorithm == "fedavg":
                idxs_users = random.sample(range(int(args.num_users)), int(args.num_users * args.frac))
              
                logger.info('user %s', sorted(idxs_users))

                for i in range(len(idxs_users)):
                    test_count[idxs_users[i]][0] += 1 

            elif args.algorithm == "fedbn2":
                idxs_users = random.sample(range(50), 20)
                # idxs_users = random.sample(range(25), 10)
                # other = random.sample(range(25, 50), 10)
                # idxs_users.extend(other)
                
                logger.info('user %s', sorted(idxs_users))
            elif iter == 1:
                idxs_users = random.sample(range(int(args.num_users*args.ratio)), 5)
                other = random.sample(range(int(args.num_users*args.ratio), 50), 5 )
                idxs_users.extend(other)


            for i in range(len(idxs_users)):
                test_count[idxs_users[i]][0] += 1 

            for idx in idxs_users:
                
                
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], learning_rate =learning_rate)
                w = local.train(net=copy.deepcopy(net_glob).to(args.device))
                g = get_gradient(args, w_glob, w, learning_rate)
                gradient[idx] = copy.deepcopy(g)
                
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals[idx] = copy.deepcopy(w)
                    # w_locals.append(copy.deepcopy(w))
                # loss_locals.append(copy.deepcopy(loss))

            if args.algorithm == "fedsel":

                gradient['avg_grad'] = average(gradient)
                max_now = get_relation(gradient, idxs_users)
                
                w_locals, idxs_before, idxs_left, labeled, test_count = Feddel(net_glob, w_locals, gradient, idxs_users, max_now, dataset_test, test_sampler_temp, args, test_count)
                
                remove_list = Diff(idxs_before, idxs_left)

                remove_who[iter] = remove_list
                logger.info("labeled %s, remove %s ", labeled, remove_list)
                w_glob = FedAvg(w_locals, idxs_left)
                # print(w_locals.keys())
                idxs_users, node_prob, test_count = probabilistic_selection(node_prob, test_count, idxs_before, idxs_left, labeled, args.prob_ratio)
                
                logger.info("round %s, prob %s", iter, node_prob)
                logger.info("round %s, count%s", iter, test_count.values())
                
            elif args.algorithm == "fedbn2":  
                w_glob, avg_iid, avg_niid = Fedbn2(w_locals, gradient)
                logger.info("round %s, avg norm %s %s", iter, avg_iid, avg_niid)
            else:
                # gradient['avg_grad'] = average(gradient)
                # vari_norm, norm_vari  = get_norm(gradient)
             
                w_glob = FedAvg(w_locals, idxs_users)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            # loss_avg = sum(loss_locals) / len(loss_locals)
            # loss_train.append(loss_avg)

            learning_rate = max(0.995 * learning_rate, args.lr * 0.01)
        # testing

            net_glob.eval()
            acc_tr, loss_tr = test_img(net_glob, dataset_train, args, train_sampler)
           
            loss_train.append(loss_tr)
            acc, loss = test_img(net_glob, dataset_test, args, test_sampler)
            logger.info("round %s Loss: %s, Accuracy: %s ", iter, round(loss,3), "{:.2f}".format(acc))
            acc_test.append(acc)
            loss_test.append(loss)

            # vari_norm_round.append(vari_norm)
            # norm_vari_round.append(norm_vari)

            # print("round %s Loss: %s, Accuracy: %s ", iter, round(loss_avg,3), "{:.2f}".format(acc))
            # norm_iid.append(avg_iid)
            # norm_niid.append(avg_niid)
        if args.algorithm == "fedsel":
            save_obj_more(acc_test, loss_test, loss_train, "%s_%s_%s_%s_exp%s_%s_%s_labeled"%(args.algorithm, args.dataset, args.model, args.local_ep, exp, args.ratio, args.prob_ratio))
        elif args.algorithm == "fedavg":
            # save_obj_more(acc_test, loss_test, loss_train, vari_norm_round, norm_vari_round, "%s_%s_%s_%s_exp%s_%s"%(args.algorithm, args.dataset, args.model, args.local_ep, exp,  args.ratio))
            save_obj_more(acc_test, loss_test, loss_train, "%s_%s_%s_%s_exp%s_%s_%s"%(args.algorithm, args.dataset, args.model, args.local_ep, exp,  args.ratio, args.frac))
        if args.algorithm == "fedbn2":
            save_obj(acc_test, loss_test, "%s_%s_%s_%s_exp%s_%s"%(args.algorithm, args.dataset, args.model, args.local_ep, exp, args.ratio))
            save_obj(norm_iid, norm_niid, "sts_fedbn2_exp%s_%s" %(exp, args.ratio))
        # if args.algorithm == "fedavg":
        #     save_obj_more(whichnode, loss_test, loss_train, "my node_exp%s_%s" %(exp, args.model))
        
