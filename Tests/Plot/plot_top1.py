import csv
import os.path
from matplotlib import pyplot as plt
import numpy as np

n_trial = 5
top_k = 10
batch_size = 4000
# max_step = 5e6

prepath = "../"
exps = ["CartpoleNd"]
# plolicies = ["MCTS_RS","MCTS_AS","MCTS_BV","RLInter","GAInter","GAISInter","GAISNInter","GATRInter","GATRISInter"]
# plot_name = "Total"
# plolicies = ["GAInter","GAISInter","GAISNInter","GATRInter","GATRISInter","GATRISNInter"]
# plot_name = "GA"
# plolicies = ["GAInter","GANonInter","GAMeanInter","GAMeanNonInter"]
# plot_name = 'GA_max_mean'
# plolicies = ["GATRInter","GATRISInter","GATRISNInter","GATRInter_kl01","GATRISInter_kl01","GATRISNInter_kl01"]
# plot_name = "GA_kl"
plolicies = ["GATRISInter_kl01"]
plot_name = "GA_TRISInter_kl01"
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
colors = []
for i in range(len(plolicies)):
    colors.append('C'+str(i))

for exp in exps:
    plts = []
    legends = []
    fig = plt.figure(figsize=(10, 10))
    for (policy_index,policy) in enumerate(plolicies):
        print(policy)
        for trial in range(n_trial):
            print(trial)
            steps = []
            rewards = []
            file_path = prepath+exp+'/Data/AST/Lexington/'+policy+'/'+str(trial)+'/process.csv'
            if os.path.exists(file_path):
                with open(file_path) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for (i,row) in enumerate(csv_reader):
                        if i == 0:
                            entry_dict = {}
                            for index in range(len(row)):
                                entry_dict[row[index]] = index
                        else:
                            # if int(row[entry_dict["StepNum"]]) > max_step:
                                # break
                            if int(row[entry_dict["StepNum"]])%batch_size == 0:
                                steps.append(int(row[entry_dict["StepNum"]]))
                                rewards.append(max(0.0,float(row[entry_dict["reward 0"]])))
                    plot, = plt.plot(steps,rewards,color=colors[policy_index])
                    plts.append(plot)
                    legends.append(policy+' '+str(trial))

    plt.legend(plts,legends)
    plt.xlabel('Step number')
    plt.ylabel('Best reward')        
    fig.savefig(prepath+exp+'/Data/Plot/top1/'+plot_name+'_top1.pdf')
    plt.close(fig)