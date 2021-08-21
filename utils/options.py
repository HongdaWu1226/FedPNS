#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    
    # federated arguments
    parser.add_argument('--rounds', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=50, help="number of users: K")
    parser.add_argument('--ratio', type=float, default=0.5,  help="portion of iid user: ")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--varrho', type=float, default=1, help="variance of synthetic data, chosen from 0.5 or 1")
    
    parser.add_argument('--num_exp', type=int, default=1, help="rounds of training")

    # model arguments
    parser.add_argument('--algorithm', type=str, default='fedavg', help='which algorithm')
    parser.add_argument('--prob_ratio', type=float, default=2.0, help="probabiltiy decay")
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument("--num_sample", type=int, default=200, help="number of samples for each node")
    parser.add_argument('--pattern', type=str,  default='iid', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args
