#!/usr/bin/env bash



python  -u main_fed.py --algorithm fedavg --dataset mnist --model mlr --ratio 0.5 --frac 0.3 --num_exp 5 --rounds 100 --pattern iid

python  -u main_fed.py --algorithm fedavg --dataset mnist --model cnn --ratio 0.5 --frac 0.3 --num_exp 4 --rounds 200 --pattern iid
 
python  -u main_fed.py --algorithm fedavg --dataset cifar --model cnn --ratio 0.5 --frac 0.3 --num_exp 5 --rounds 200 --num_channels 3 --local_ep 5 --pattern iid

#python  -u main_fed.py --algorithm fedavg --dataset cifar --model cnn --ratio 1 --frac 0.6 --num_exp 3 --rounds 200 --num_channels 3 --local_ep 5

#python  -u main_fed.py --algorithm fedavg --dataset cifar --model cnn --ratio 1 --frac 1.0 --num_exp 3 --rounds 200 --num_channels 3 --local_ep 5















