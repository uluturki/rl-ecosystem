
import os, sys

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_cumulative_variation

'''
Plot the population dynamics for the exeperiment which aims at verifying whether agents are properly trained or not
'''

argparser = argparse.ArgumentParser()
argparser.add_argument('--log_file', type=str)
argparser.add_argument('--st', type=int, default=0)
argparser.add_argument('--ed', type=int, default=None)
args = argparser.parse_args()

log_file = args.log_file

plot_cumulative_variation(args.log_file, args.st, args.ed)
