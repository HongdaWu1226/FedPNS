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

labels = ['FedPNS', 'FedAvg' ]

iid_list = [10]
niid_list = [10]

model = [ 'cnn']
ratio = [0.2, 0.3,0.5]
pkl_path = ["varrho=1/", "varrho=0.5/"]

ylabel = ['Training Loss', 'Test Accuracy']
num_exp = [5, 5]
num_round = 200

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
  
    for row in range(2):
       
        column = -1
        for i in range(2):
            for j in range(3):
                column += 1
                fedavg_data = {}
                fedadp_data = {}
                
                for exp in range(1,num_exp[i]+1):
                    fedadp_data[exp] = load_obj('fedsel_20_exp%s_%s' %(exp, ratio[j]), pkl_path[i] )
                
                for exp in range(1,num_exp[i]+1):
                    fedavg_data[exp] = load_obj('fedavg_20_exp%s_%s' %(exp, ratio[j]), pkl_path[i])
                
                # if args.matrice == "acc":
                
                overall_avg = []  
                for k in range(1,num_exp[i]+1):
                    overall_avg.extend(fedadp_data[k][1-row])
                
                temp_adp = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp[i])])
                acc_adp = np.mean(temp_adp, axis=0)
                var_set[row][column].plot(list(range(num_round)), acc_adp, color='c', linewidth = '2',label = labels[0])

                overall_avg = []  
                for k in range(1,num_exp[i]+1):
                    overall_avg.extend(fedavg_data[k][1-row])
                
                temp_avg = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp[i])])
                acc_avg = np.mean(temp_avg, axis=0)
                var_set[row][column].plot(list(range(num_round)), acc_avg, '--',color='#F97306', linewidth = '2', label = labels[-1])


                var_set[row][column].set_ylabel(ylabel[row], fontsize=16)
                var_set[row][column].set_xlabel('Communication Round', fontsize=16)

                var_set[row][column].spines['right'].set_visible(False)
                var_set[row][column].spines['top'].set_visible(False)
           
    
    var_set[0][0].set_title('$\sigma = 0.2, \\varrho = 1$',fontweight="bold", size=18)
    var_set[0][1].set_title('$\sigma = 0.3, \\varrho = 1$', fontweight="bold", size=18)
    var_set[0][2].set_title('$\sigma = 0.5, \\varrho = 1$', fontweight="bold", size=18)

    var_set[0][3].set_title('$\sigma = 0.2, \\varrho = 0.5$',fontweight="bold", size=18)
    var_set[0][4].set_title('$\sigma = 0.3, \\varrho = 0.5$', fontweight="bold", size=18)
    var_set[0][5].set_title('$\sigma = 0.5, \\varrho = 0.5$', fontweight="bold", size=18)

    fig.subplots_adjust(top=0.9, left=0.08, right=0.9, bottom=0.08)  # create some space below the plots by increasing the bottom-value
    var_set.flatten()[-2].legend(frameon=False, loc='upper center', prop={'size': 18}, bbox_to_anchor=(-1.2,2.5), ncol=2)
    # fig.tight_layout()
    fig_path = ""

    plt.savefig(fig_path + "synth" + ".eps", format='eps', dpi=1200)

            
    
    

if __name__ == "__main__":

    main()



