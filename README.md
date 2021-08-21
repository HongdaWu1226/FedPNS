# Node Selection Toward Faster Convergence for Federated Learning on Non-IID Data (submitted to IEEE TNSE)


This repository contains the code and experiments for the paper: 

Node Selection Toward Faster Convergence for Federated Learning on Non-IID Data 
>   Authors: Hongda Wu, Ping Wang <br>
Full paper: https://arxiv.org/pdf/2105.07066.pdf

Federated Learning is a distributed learning paradigm that enables a large number of resource-limited nodes to collaboratively train a model without data sharing. The non-independent-and-identically-distributed (non-i.i.d.) data samples invoke discrepancy between global and local objectives, making the FL model slow to converge. In this work, we propose FedPNS, a probabilistic node selection design, to handle the convergence slowness of heterogeneous data. The convergence rate improvement of FedPNS over the commonly adopted Federated Averaging (FedAvg) algorithm is analyzed theoretically and verified empirically. 

This repository contains a set of detailed empirical evaluation across a suite of federated datasets. We show that FedPNS preferentially choose the nodes with higher contribution so as to improves the convergence rate, as compared with FedAvg.



# General Guidelines:
* We use different models (MLR and CNN) on synthetic data and real datasets (MNIST, CIFAR10) for experiments. 
* The data heterogeneity is controlled two parameters, for synthetic dataset, _varrho_ (0.5 or 1, represents the variance of data samples in each non-i.i.d. node) and _sigma_ (0.2, 0.3 or 0.5, indicates the ratio of i.i.d. nodes among all nodes). In real datasets, the data heterogeneity is controlled by _rho_ and _sigma_, where _rho_ (the number of labels that data samples on non-i.i.d. nodes belongs to) replaces the _varrho_ in Synthetic data.
* The code can be run on cpu, and is cocompatible with GPU.


# Preparation: 
## Downloading dependencies

      pip3 install -r requirements.txt

## Synthetic dataset
(1). Run the instructions as follows, the Training Loss, Test Accuracy will be stored as .pkl file (each of the above results is in _list_ format) in self-defined path (in `mclr.py`, e.g., "./result/fig2_syn/varrho=1/"). The data heterogeneity is controlled by _varrho_ (0.5 or 1) and _ratio_ (0.2, 0.3, 0.5). For example, one can run to generated the result when sigma = 0.5, varrho  = 1, 

      python  -u mclr.py --algorithm fedavg --model mlr --frac 0.2 --num_exp 10 --rounds 200  --num_sample 1000 --local_ep 20 --varrho 1 --ratio 0.5 
      python  -u mclr.py --algorithm fedsel --model mlr --frac 0.2 --num_exp 10 --rounds 200  --num_sample 1000 --local_ep 20 --varrho 1 --ratio 0.5

(2). To reproduce the Fig. 2 in manuscript, go to the path "./result/fig2_syn/" and run `all.py`. The plot shows the averaged result of multiple experiments.


## Real dataset (MNIST, CIFAR10)
(1). Run the instructions as follows, the Training Loss, Test Accuracy, Test loss will be stored as .pkl file (each of the above results is in _list_ format) in self-defined path (in `main_fed.py`, e.g., "./result/fig3_mlr_cnnM/varrho=1/"). The data heterogeneity is controlled by _rho_ (1 or 2) and _ratio_ (0.2, 0.3, 0.5). 

For example, one can run to generated the result when sigma = 0.5, rho  = 1 on MNIST dataset using MLR model

      python  -u main_fed.py --algorithm fedavg --model mlr --pattern non-i.i.d. --frac 0.2 --num_exp 10 --rounds 100  --num_sample 200 --local_ep 1 --ratio 0.5 
      python  -u main_fed.py --algorithm fedsel --model mlr --pattern non-i.i.d. --frac 0.2 --num_exp 10 --rounds 100  --num_sample 200 --local_ep 1  --ratio 0.5

one can run to generated the result when sigma = 0.5, rho  = 1 on CIFAR10 dataset using MLR model

      python  -u main_fed.py --algorithm fedavg --model mlr --pattern non-i.i.d. --frac 0.2 --num_exp 10 --rounds 200  --num_sample 200 --local_ep 5 --ratio 0.5 -- num_channels 3
      python  -u main_fed.py --algorithm fedsel --model mlr --pattern non-i.i.d. --frac 0.2 --num_exp 10 --rounds 200  --num_sample 200 --local_ep 5  --ratio 0.5 -- num_channels 3

In the cases rho  = 2, modify the function `noniid ` in `utils/sampling.py` by concatenating two shards. 


(2). To reproduce the Fig. 3 or 4 in manuscript, go to the path "./result/fig3_mlr_cnnM/" or "./result/fig4_cifar/", and run `all.py`. The plot shows the averaged result of multiple experiments.

## To reproduce Fig. 6, 

1) modify the "--algorithm" part of above script as "--algorithm bn2" and get the results, e.g., 

      python  -u main_fed.py --algorithm fedbn2 --model cnn --pattern non-i.i.d. --num_exp 10 --rounds 200  --num_sample 200 --local_ep 1 --ratio 0.5   
      
 2) go to "./result/fig6_norm_comp/" to get the results in Fig. 6.

## Effect of hyper-parameters:

* We use learning_rate =0.01 and learning_rate_decay (per round) = 0.995 through all experiments.
* The choice of _alpha_ and _beta_ are fixed as 2 and 0.7, respectively except for the Fig. 5. To reproduce Fig. 5, one can modify the `probabilistic_selection` in `models/func.py`.

# References
See our [FedPNS](https://arxiv.org/pdf/2105.07066) paper for more details as well as all references.

