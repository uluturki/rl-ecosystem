import os, sys

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import spline

argparser = argparse.ArgumentParser()
argparser.add_argument('--log_file', type=str)
args = argparser.parse_args()

log_file = args.log_file

# ['Episode', '004', 'Step', '000', 'Reward', '10.000', 'num_agents', '186', 'num_preys', '86', 'num_predators', '100']
# Step	000	Reward	17.500	num_agents	750	num_preys	250	num_predators	500

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


#st =5000
#st = 0
#ed = len(predator_num)
st = 2000
#ed = len(predator_num)
ed = len(predator_num)

x = range(len(prey_num))
sns.set_style("darkgrid")
plt.plot(x[st:ed], predator_num[st:ed])
plt.plot(x[st:ed], prey_num[st:ed])
plt.legend(['predators', 'preys'])
plt.show()

plt.savefig(os.path.join(os.path.dirname(log_file),'agent_num_plot.png'))

