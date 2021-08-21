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

labels = ['Optimal Aggregation', 'FedAvg' ]
labels_prun = ['fedadp_5iid_5niid_0.8_prun' , 'fedadp_5iid_5niid_current_prun','fedadp_5iid_5niid']
# iid_list = [5, 6, 2, 8]
# niid_list = [10 - x for x in iid_list]
iid_list = [10]
niid_list = [10]
prob_ratio = [0.1]
model = [ 'cnn']

num_exp = 10
num_exp_3 = 3
num_round = 50

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

    fedavg_data = {}
    fedadp_data = {}
    feddel_data = {}
    remove_node = {}

    
    # for exp in range(1,num_exp+1):
    #     remove_node[exp] = load_obj('remove node_exp%s' %(exp))
    # print(remove_node[2][0])

    for exp in range(1,num_exp+1):
        fedadp_data[exp] = load_obj('feddel_mnist_%s_1_exp%s' %(model[0], exp))
    
   
    for exp in range(1,num_exp+1):
        fedavg_data[exp] = load_obj('fedavg_mnist_%s_1_exp%s' %(model[0],exp))
    
    if args.matrice == "acc":

        overall_avg = []  
        for k in range(1,num_exp+1):
            # print(fedadp_data[k][0])
            overall_avg.extend(fedadp_data[k][0])
        
        temp_adp = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
        acc_adp = np.mean(temp_adp, axis=0)
        # print(acc_adp)
        result_plt(acc_adp, labels[0]) 

        overall_avg = []  
        for k in range(1,num_exp+1):
            overall_avg.extend(fedavg_data[k][0])
        
        temp_avg = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
        acc_avg = np.mean(temp_avg, axis=0)
        # print(acc_avg)
        result_plt(acc_avg, labels[-1])  
        ylabel = "Testing Accuracy"

    elif args.matrice == "loss":

        overall_avg = []  
        for k in range(1,num_exp+1):
            # print(fedadp_data[k][0])
            overall_avg.extend(fedadp_data[k][1])
        
        temp_adp = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
        acc_adp = np.mean(temp_adp, axis=0)
        # print(acc_adp)
        plt.plot(list(range(num_round)), acc_adp, color='#069AF3', linewidth = '1.5', label = labels[0])
        overall_avg = []  
        for k in range(1,num_exp+1):
            overall_avg.extend(fedavg_data[k][1])
        
        temp_avg = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(num_exp)])
        acc_avg = np.mean(temp_avg, axis=0)
        plt.plot(list(range(num_round)), acc_avg, '--', color='#F97306', linewidth = '1.5',label = labels[-1])

        

        plt.xlabel("Communication Round", fontsize=13)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.ylabel( "Training Loss", fontsize=13)
        
        plt.legend(frameon=False, loc=7, prop={'size': 10})

    elif args.matrice == "sts":

        x =  load_obj('observe node_exp10' )

        # print(x[0])
        overall_avg = []
        for i in range(3):
            temp = []
            for j in range(50):
                # print(x[0][j][i])
                temp.append(x[0][j][i])
            
            overall_avg.extend(temp)

        data = np.array([overall_avg[num_round*i:num_round*(i+1)] for i in range(3)])
        
        # print(data[0])
        # plt.figure()
        # plt.subplot()

        # fig, ax = plt.subplots(nrows=2, ncols=1)

        

        label = ['Selected', 'Labeled', 'Excluded']
        
        index = np.arange(0, 25, 1)
        index_2 = np.arange(25, 50, 1)
        # plt.hist()
        color_index= ['lightgray','lightsteelblue','springgreen']
       
        plt.subplot(2,1,1)
        for i in range(3):
            j = i+1
            #index+2:是起始坐标点    #width是bar的宽度
            # print(data[i])
            plt.bar(index, data[i][:25],width=0.6,color=color_index[i], label= label[i])
        plt.xticks(index)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=8)

        plt.subplot(2,1,2)
        for i in range(3):
            j = i+1
            #index+2:是起始坐标点    #width是bar的宽度
            plt.bar(index_2, data[i][25:],width=0.6,color=color_index[i], label= label[i])
        plt.xticks(index_2)
        plt.yticks([0,15, 5, 10])
        plt.legend(loc='best', prop={'size': 7})
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=8)
        
        # plt.gca().spines['top'].set_visible(False)
       
        #     plt.hist(data[i], index, alpha = 0.5)
        # plt.hist(data[0], index, alpha = 0.5)
        # plt.hist(data[1], index, alpha = 0.5)
 

   

    fig_path = ""
   
    plt.savefig(fig_path + "%s_com_%siid_%s" %(args.matrice, str(iid_list[0]), model[0]) + ".eps", format='eps', dpi=1200)

    
    

if __name__ == "__main__":

    main()



