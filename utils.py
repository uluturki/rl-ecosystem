import os, sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dynamics(log_file, st):
    prey_num = []
    predator_num = []
    with open(log_file)as fin:
        for line in fin:
            line = line.split()
            if len(line) == 12:
                prey_num.append(int(line[9]))
                predator_num.append(int(line[11]))
            elif len(line) == 10:
                prey_num.append(int(line[7]))
                predator_num.append(int(line[9]))
    ed = len(predator_num)

    x = range(len(prey_num))
    plt.figure()
    sns.set_style("darkgrid")
    plt.plot(x[st:ed], predator_num[st:ed])
    plt.plot(x[st:ed], prey_num[st:ed])
    plt.legend(['predators', 'preys'])
    plt.show()

    plt.savefig(os.path.join(os.path.dirname(log_file),'agent_num_plot.png'))

