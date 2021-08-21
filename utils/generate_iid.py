import json, math, os, sys
import numpy as np
import random
from tqdm import trange



def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def generate_synthetic_iid(alpha, beta, iid, nr_sample, num_users, ratio):
    dimension = 60
    NUM_CLASS = 10  
    NUM_USER = int(num_users*ratio)
    
    samples_per_user = [nr_sample for i in range(NUM_USER)]
   
    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]

    #### define some eprior ####
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        mean_x[i] = np.zeros(dimension)

    W = np.random.normal(0, 1, (dimension, NUM_CLASS))
    b = np.random.normal(0, 1,  NUM_CLASS)

    for i in range(NUM_USER):
        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

    return X_split, y_split


def generate_iid_test(alpha, beta, iid, nr_sample, num_users, ratio):
    dimension = 60
    NUM_CLASS = 10  
        
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    X, y = generate_synthetic_iid(0, 0, 0, nr_sample, num_users, ratio)

    for i in range(1):
        # print("iid", i)

        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(1 * num_samples)
        
        test_data['user_data'][i] = {'x': X[i][:], 'y': y[i][:]}
        test_data['num_samples'].append(train_len)
        

    return test_data


def generate_synthetic_niid(alpha, varrho, iid, nr_sample, num_users, ratio):

    dimension = 60
    NUM_CLASS = 10
    NUM_USER = int(num_users*(1-ratio))
    # print(NUM_USER)
    samples_per_user = [nr_sample for i in range(NUM_USER)]
    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]


    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, varrho, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        mean_x[i] = np.random.normal(B[i], 1, dimension)
        # print(mean_x[i])

    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1,  NUM_CLASS)

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()



    return X_split, y_split


def data(nr_sample, num_users, varrho, ratio):


    X, y = generate_synthetic_iid(0, 0, 1, nr_sample, num_users, ratio)
    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}


    us_list = []
    for i in range(int(num_users*ratio)):
        # print("iid", i)
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.6 * num_samples)
        test_len = num_samples - train_len
        
        train_data['users'].append(i) 
        train_data['user_data'][i] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)

        test_data['user_data'][i] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        us_list.append(i)


    
    M, n = generate_synthetic_niid(0, varrho, 0, nr_sample, num_users, ratio)
    
    for i in range(int(num_users*(1-ratio))):
        
         
        combined = list(zip(M[i], n[i]))
        random.shuffle(combined)
        M[i][:], n[i][:] = zip(*combined)
        num_samples = len(M[i])
        train_len = int(0.6 * num_samples)
        
        train_data['users'].append(i+int(num_users*ratio)) 
        train_data['user_data'][i+int(num_users*ratio)] = {'x': M[i][:train_len], 'y': n[i][:train_len]}
        train_data['num_samples'].append(train_len)
        us_list.append(i+int(num_users*ratio))


    return train_data, test_data, us_list
  


