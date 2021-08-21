import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle



def load_obj(name, pkl_path):
    # pkl_path = ""
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

labels = ['$\\beta = 0.7, \\alpha=2$', '$\\beta = 0.5, \\alpha=2$', '$\\beta = 0.8, \\alpha=2$','FedAvg' ]
labels_prun = ['$\\beta = 0.7, \\alpha=2$', '$\\beta = 0.7, \\alpha=1$', '$\\beta = 0.7, \\alpha=3$','FedAvg']


iid_list = [10]
niid_list = [10]

model = [ 'cnn']
ratio = [ 0.3, 0.5]
prob_ratio = [1.0, 3.0, 2.0, 2.0 ]

pkl_path = ["alpha/", "beta/"]
indx = [0, 2]
num_exp = 5
num_round = 200
num_round_3 = 300

def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run figure plot"
    )
    parser.add_argument("--matrice", type=str, choices=matrices, default="loss", help = "result matrices")
    parser.add_argument("--iid", type=int, default=5, help="number of nodes")
    parser.add_argument("--training_rounds", type=int, default = 50, help= "number of training rounds")

    args = parser.parse_args(args=args)
    return args



def main():

    args = define_and_get_arguments()

    
    fig, var_set = plt.subplots(2, 2, figsize=(13,8))
    # var_set = [[ax1, ax2, ax3,ax4], [bx1, bx2, bx3,bx4]]
    
    for row in range(2):
        column = -1
        for i in range(1):
            for j in range(2):
                column += 1
                fedavg_data = {}
                fedadp_data = {}
                fedadp_data_2 = {}
                fedadp_data_3 = {}
                for exp in range(1,num_exp+1):

                    if row == 0:
                    
                        label = labels_prun
                        
                        fedadp_data[exp] = load_obj('fedsel_mnist_%s_1_exp%s_%s_%s_labeled' %(model[0], exp,  ratio[j], prob_ratio[-1]), pkl_path[row] )
                        fedavg_data[exp] = load_obj('fedavg_mnist_%s_1_exp%s_%s' %(model[0],exp, ratio[j]), pkl_path[row])
                        fedadp_data_2[exp] = load_obj('fedsel_mnist_%s_1_exp%s_%s_%s_labeled' %(model[0], exp,  ratio[j], prob_ratio[0]), pkl_path[row] )
                        fedadp_data_3[exp] = load_obj('fedsel_mnist_%s_1_exp%s_%s_%s_labeled' %(model[0], exp,  ratio[j], prob_ratio[1]), pkl_path[row] )                       
                    else:
                        label = labels
                        fedadp_data[exp] = load_obj('fedsel_mnist_%s_1_exp%s_%s_%s_labeled' %(model[0], exp,  ratio[j], prob_ratio[-1]), pkl_path[row] )
                        fedavg_data[exp] = load_obj('fedavg_mnist_%s_1_exp%s_%s' %(model[0],exp, ratio[j]), pkl_path[row])
                        fedadp_data_2[exp] = load_obj('fedsel_mnist_%s_1_exp%s_%s_%s_labeled_0.5' %(model[0], exp,  ratio[j], prob_ratio[-1]), pkl_path[row] )
                        fedadp_data_3[exp] = load_obj('fedsel_mnist_%s_1_exp%s_%s_%s_labeled_0.8' %(model[0], exp,  ratio[j], prob_ratio[-1]), pkl_path[row] )
                
                overall_avg = []  
                for k in range(1,num_exp+1):
                    overall_avg.extend(fedadp_data[k][0])
                
                temp_adp = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
                acc_adp = np.mean(temp_adp, axis=0)
                var_set[row][column].plot(list(range(num_round)), acc_adp, color='c', linewidth = '2',label = label[0])
                
                overall_avg = []  
                for k in range(1,num_exp+1):
                    overall_avg.extend(fedadp_data_2[k][0])
                
                temp_adp = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
                acc_adp = np.mean(temp_adp, axis=0)
                var_set[row][column].plot(list(range(num_round)), acc_adp, linewidth = '2',label = label[1])

                overall_avg = []  
                for k in range(1,num_exp+1):
                    overall_avg.extend(fedadp_data_3[k][0])
                
                temp_adp = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
                acc_adp = np.mean(temp_adp, axis=0)
                var_set[row][column].plot(list(range(num_round)), acc_adp, color='springgreen', linewidth = '2',label = label[2])


                overall_avg = []  
                for k in range(1,num_exp+1):
                    overall_avg.extend(fedavg_data[k][0])
                
                temp_avg = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
                acc_avg = np.mean(temp_avg, axis=0)
                # print(acc_avg)
                var_set[row][column].plot(list(range(num_round)), acc_avg, '--',color='#F97306', linewidth = '2', label = label[-1])

                var_set[row][column].set_ylim([70, 95])

                var_set[row][column].set_ylabel('Test Accuracy', fontsize=16)
                var_set[row][column].set_xlabel('Communication Round', fontsize=16)

                var_set[row][column].spines['right'].set_visible(False)
                var_set[row][column].spines['top'].set_visible(False)
            # plt.ylabel("Testing Accuracy", fontsize=12)
            # plt.xlabel("Communication Rounds", fontsize=12)

            # x1,x2,y1,y2 = plt.axis()
            # plt.axis((x1,x2,50,95))
                # legend(frameon=False, loc='lower center', ncol=2)
    # plt.legend(frameon=False, loc=7, prop={'size': 10}, ncol=2)
        var_set[row][0].legend(frameon=False, loc=2, prop={'size': 12})
        var_set[row][1].legend(frameon=False, loc=2, prop={'size': 12})

    var_set[0][0].set_title('$\sigma = 0.3, \\rho=1 $', fontweight="bold", size=18)
    var_set[0][1].set_title('$\sigma = 0.5, \\rho=1 $', fontweight="bold", size=18)


    fig_path = ""

    plt.savefig(fig_path + "alpha_beta"  + ".eps", format='eps', dpi=1200)
    # plt.close()


            
    
    

if __name__ == "__main__":

    main()



