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

def plot_circle(log_file, st):
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
    x = range(len(predator_num[st:]))
    length = len(predator_num[st:])

    predator_num_avg = []
    prey_num_avg = []

    for i in range(0, length, 10):
        predator_tot = 0
        prey_tot = 0
        for j in range(i, min(i + 10, len(x))):
            predator_tot += predator_num[j]
            prey_tot += prey_num[j]

        predator_tot = 1. * predator_tot / 10.
        prey_tot = 1. * prey_tot / 10.
        predator_num_avg.append(predator_tot)
        prey_num_avg.append(prey_tot)

    plt.figure(figsize=(8, 6))
    plt.plot(predator_num_avg, prey_num_avg, label='number')
    plt.xlabel('predator number')
    plt.ylabel('prey number')
    plt.legend(['predator number', 'prey number'], loc='upper left')
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(log_file),'circle.png'))
