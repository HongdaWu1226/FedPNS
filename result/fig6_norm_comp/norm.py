import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle



def load_obj(name):
    pkl_path = ""
    with open(pkl_path + name + ".pkl", 'rb') as f:
        return pickle.load(f)

def load_prun_obj(name):
    pkl_path = ""
    with open(pkl_path + name + ".pkl", 'rb') as f:
        return pickle.load(f)

def result_plt(results, label):
    # lists = sorted(results.items())
    # x, y = zip(*lists)
    plt.plot(results, label = label)


matrices = ['acc', 'loss']    

# labels = ['fedavg_5iid_5niid', 'fedavg_6iid_4niid', 'fedavg_2iid_8niid', 'fedavg_8iid_2niid' ]
# labels_prun = ['fedavg_5iid_5niid_prun', 'fedavg_6iid_4niid_prun', 'fedavg_8iid_2niid_prun']

labels = ['prob_select', 'bn2 [5]', 'random' ]
labels_prun = ['i.i.d.', 'non-i.i.d.']
# iid_list = [5, 6, 2, 8]
# niid_list = [10 - x for x in iid_list]
iid_list = [10]
niid_list = [10]
prob_ratio = [0.1]
model = [ 'cnn']

num_exp = 5
num_exp_3 = 3
num_round = 200

def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run figure plot"
    )
    parser.add_argument("--matrice", type=str, choices=matrices, default="acc", help = "result matrices")
    parser.add_argument("--iid", type=int, default=5, help="number of nodes")
    parser.add_argument("--training_rounds", type=int, default = 50, help= "number of training rounds")

    args = parser.parse_args(args=args)
    return args




def main():

    args = define_and_get_arguments()

    fedavg_data = {}
    fedadp_data = {}
    feddel_data = {}
    remove_node = {}

    ax = plt.subplot(111)
    # x =  load_obj('fedbn2_cifar_cnn_5_exp4' )
    # print(x[0])
    for exp in range(1,num_exp+1):
        remove_node[exp] = load_obj('sts_fedbn2_exp%s_0.5' %(exp))


    if args.matrice == "acc":

  
        overall_avg = []

        for k in range(1,num_exp+1):
            # print(fedadp_data[k][0])
            overall_avg.extend(remove_node[k][0])

        temp_adp = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
        acc_adp = np.mean(temp_adp, axis=0)
        # print(acc_adp)
        # result_plt(acc_adp, labels_prun[0])
        ax.plot(list(range(num_round)), acc_adp, linewidth = '2',label = labels_prun[0])

        overall_avg = []
        for k in range(1,num_exp+1):
            # print(fedadp_data[k][0])
            overall_avg.extend(remove_node[k][1])

        temp_adp = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
        acc_adp = np.mean(temp_adp, axis=0)
        # print(acc_adp)
        ax.plot(list(range(num_round)), acc_adp, linewidth = '2',label = labels_prun[-1])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.ylabel('Norm of Gradient', fontsize=13)
        plt.xlabel('Communication Round', fontsize=13)
        plt.legend(frameon=False, loc=7, prop={'size': 10})

        



    
    fig_path = ""
   
    plt.savefig(fig_path + "norm_gradient" + ".eps", format='eps', dpi=1200)

    
    

if __name__ == "__main__":

    main()



