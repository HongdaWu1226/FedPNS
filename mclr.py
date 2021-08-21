import numpy as np
import tensorflow as tf
import logging
import copy
import torch
import random
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from tqdm import trange
from utils.generate_iid import data
from models.Update import LocalUpdate
from utils.options import args_parser
from models.Fed import FedAvg
from models.func import save_obj
from models.func import get_gradient
from models.func import get_relation
from models.func import Diff
from models.func import probabilistic_selection
from models.Fed import Feddel_syn
from models.Fed import average

class MLR(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLR, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = x.reshape(-1, 60)
        # out = F.sigmoid(self.linear(xb))
        
        out = self.sigmoid(self.linear(x))
        return F.log_softmax(out, dim=1)

def create_model (model, num_classes, device, img_size):
    if model == 'mlr':
        
        net_glob = MLR(dim_in=img_size, dim_out=num_classes).to(device)

    return net_glob

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


def local_training(args, net, data, lr ):
    optimizer = torch.optim.SGD(net.parameters(), lr)
    loss_func = nn.CrossEntropyLoss()
    for _ in range(args.local_ep):
        batch_loss = []
        for X, y in batch_data(data, args.local_bs):
        
            net.zero_grad()
            log_probs = net(X)
            
            loss = loss_func(log_probs, y.long())
            # print(loss)
            loss.backward()
            optimizer.step()
        
            batch_loss.append(loss.item())

        # epoch_loss.append(sum(batch_loss)/len(batch_loss))
    return net.state_dict()

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

if __name__ == "__main__":
    logger = logging.getLogger("mclr")
    logger.setLevel(level=logging.DEBUG)
    args = args_parser()
    logging.basicConfig(filename = "./result/fig2_syn/varrho=1/" + "%s_%s_%s"
                        %(args.algorithm, args.local_ep, args.ratio) + ".txt")
    # parse args
    
    args = args_parser()
    device = torch.device('cpu')
    
    learning_rate = args.lr


    for exp in tqdm(range(1, args.num_exp+1)):
        dataset_train, test_data, us_list = data(args.num_sample,args.num_users, args.varrho, args.ratio)

        logger.info("--------------Experiment---------- %s/%s", exp, args.num_exp)
        net_glob = create_model('mlr', 10, device, 60)

        net_glob.train()
        w_glob = net_glob.state_dict()

        acc_test = []
        loss_test = []
        node_prob = {}
        test_count = {}
        for i in range(args.num_users):
            node_prob[i] = 1 / args.num_users
            tupe = []
            for j in range(3):
                tupe.append(0)
                test_count[i] = tupe
        whichnode = {}
        remove_who = {}

        for iter in range(1, args.rounds+1):

            if args.algorithm == "fedavg":
                idxs_users = random.sample(range(50), 10)

            elif iter == 1:
                idxs_users = random.sample(range(int(args.num_users*args.ratio)), 5)
                other = random.sample(range(int(args.num_users*args.ratio), 50), 5 )
                idxs_users.extend(other)
            for i in range(len(idxs_users)):
                test_count[idxs_users[i]][0] += 1 
            # print(idxs_users)
            w_locals = {}
            gradient = {}

            for idx in idxs_users:
                w = local_training(args, net_glob, dataset_train['user_data'][idx], learning_rate)
                w_locals[idx] = copy.deepcopy(w)

                if args.algorithm != "fedavg":
                    g = get_gradient(args, w_glob, w, learning_rate)
                    gradient[idx] = copy.deepcopy(g)


            if args.algorithm == "fedsel":

                gradient['avg_grad'] = average(gradient)
                max_now = get_relation(gradient, idxs_users)

                w_locals, idxs_before, idxs_left, labeled, test_count = Feddel_syn(net_glob, w_locals, gradient, idxs_users, max_now, test_data, args, test_count)
                
                remove_list = Diff(idxs_before, idxs_left)

                remove_who[iter] = remove_list
                logger.info("labeled %s, remove %s ", labeled, remove_list)
                # logger.info('which node %s', sorted(idxs_before))
                # print('final %s', sorted(idxs_left))
                w_glob = FedAvg(w_locals, idxs_left)
                # print(w_locals.keys())
                idxs_users, node_prob, test_count = probabilistic_selection(node_prob, test_count, idxs_before, idxs_left, labeled, args.prob_ratio)
                
                logger.info("round %s, prob %s", iter, node_prob)
                logger.info("round %s, count%s", iter, test_count.values())
            else:
                w_glob = FedAvg(w_locals, idxs_users)

            learning_rate = max(0.995 * learning_rate, args.lr * 0.01)

            net_glob.load_state_dict(w_glob)
            net_glob.eval()
         
            acc, loss = test(net_glob, test_data, args)
            logger.info("round %s Loss: %s, Accuracy: %s ", iter, round(loss,3), "{:.2f}".format(acc))
            acc_test.append(acc)
            loss_test.append(loss)

        save_obj(acc_test, loss_test, "%s_%s_exp%s_%s"%(args.algorithm, args.local_ep, exp, args.ratio))

