
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

labels = ['FedPNS', 'FedAvg' ]
labels_prun = ['fedadp_5iid_5niid_0.8_prun' , 'fedadp_5iid_5niid_current_prun','fedadp_5iid_5niid']
# iid_list = [5, 6, 2, 8]
# niid_list = [10 - x for x in iid_list]
iid_list = [10]
niid_list = [10]
model = [ 'mlr', 'cnn']
ratio = [0.2, 0.3, 0.5]
prob_ratio = [2.0, 2.0, 0.1, 0.1]
pkl_path = [ "mlr/1label/", "mlr/2label/", "cnn_design/1label/", "cnn_design/2label/"]
indx = [0, 2]
num_exp = [10, 5]
num_round = [100, 200]

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

    
    fig, var_set = plt.subplots(2, 6, figsize=(24,7))
    # var_set = [[ax1, ax2, ax3,ax4], [bx1, bx2, bx3,bx4]]
    
    for row in range(2):
        column = -1
        for i in range(2):
            for j in range(3):
                column += 1
                fedavg_data = {}
                fedadp_data = {}
                print(pkl_path[2*row + i])
                for exp in range(1,num_exp[row]+1):
                    fedadp_data[exp] = load_obj('fedsel_mnist_%s_1_exp%s_%s_%s_labeled' %(model[row], exp,  ratio[j], prob_ratio[row]), pkl_path[2*row + i] )
                
                for exp in range(1,num_exp[row]+1):
                    fedavg_data[exp] = load_obj('fedavg_mnist_%s_1_exp%s_%s' %(model[row],exp, ratio[j]), pkl_path[2*row + i])
                
                # if args.matrice == "acc":
               
                overall_avg = []  
                for k in range(1,num_exp[row]+1):
                    overall_avg.extend(fedadp_data[k][1])
                
                temp_adp = np.array([overall_avg[num_round[row]*i:num_round[row]*(i+1)] for i in range(num_exp[row])])
                acc_adp = np.mean(temp_adp, axis=0)
                var_set[row][column].plot(list(range(num_round[row])), acc_adp, color='c', linewidth = '2',label = labels[0])

                overall_avg = []  
                for k in range(1,num_exp[row]+1):
                    overall_avg.extend(fedavg_data[k][1])
                
                temp_avg = np.array([overall_avg[num_round[row]*i:num_round[row]*(i+1)] for i in range(num_exp[row])])
                acc_avg = np.mean(temp_avg, axis=0)
                # print(acc_avg)
                var_set[row][column].plot(list(range(num_round[row])), acc_avg, '--',color='#F97306', linewidth = '2', label = labels[-1])

                # var_set[row][column].set_ylim([70, 95])

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
    var_set[0][0].set_title('$\sigma = 0.2, \\rho = 1$',fontweight="bold", size=18)
    var_set[0][1].set_title('$\sigma = 0.3, \\rho = 1$', fontweight="bold", size=18)
    var_set[0][2].set_title('$\sigma = 0.5, \\rho = 1$', fontweight="bold", size=18)

    var_set[0][3].set_title('$\sigma = 0.2, \\rho = 2$',fontweight="bold", size=18)
    var_set[0][4].set_title('$\sigma = 0.3, \\rho = 2$', fontweight="bold", size=18)
    var_set[0][5].set_title('$\sigma = 0.5, \\rho = 2$', fontweight="bold", size=18)


    fig.subplots_adjust(top=0.9, left=0.08, right=0.9, bottom=0.08)  # create some space below the plots by increasing the bottom-value
    var_set.flatten()[-2].legend(frameon=False, loc='upper center', prop={'size': 18}, bbox_to_anchor=(-1.2,2.5), ncol=2)

    fig_path = ""

    plt.savefig(fig_path + "what_%s_com_%s" %(args.matrice, model[0]) + ".eps", format='eps', dpi=300)
    # plt.close()

    # elif args.matrice == "loss":

    #     overall_avg = []  
    #     for k in range(1,num_exp+1):
    #         # print(fedadp_data[k][0])
    #         overall_avg.extend(fedadp_data[k][1])
        
    #     temp_adp = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
    #     acc_adp = np.mean(temp_adp, axis=0)
    #     # print(acc_adp)
    #     result_plt(acc_adp, labels[0]) 

    #     overall_avg = []  
    #     for k in range(1,num_exp+1):
    #         overall_avg.extend(fedavg_data[k][1])
        
    #     temp_avg = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
    #     acc_avg = np.mean(temp_avg, axis=0)
    #     # print(acc_avg)
    #     result_plt(acc_avg, labels[-1])  
    #     ylabel = "Training Loss"

    #     plt.xlabel("Communication Rounds", fontsize=12)
    #     if args.matrice == "acc":
    #         x1,x2,y1,y2 = plt.axis()
    #         plt.axis((x1,x2,0,95))

    #     # else:
    #     #     x1,x2,y1,y2 = plt.axis()
    #     #     plt.axis((x1,x2,0,0.01))
        
    #     plt.ylabel(ylabel, fontsize=12)
    #     plt.xticks(fontsize=10)
    #     plt.yticks(fontsize=10)
    #     plt.legend(loc=7, prop={'size': 10})


   

            
    
    

if __name__ == "__main__":

    main()



