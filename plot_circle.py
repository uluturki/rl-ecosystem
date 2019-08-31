import os, sys

import matplotlib.pyplot as plt
import argparse
import numpy as np
import seaborn as sns
from utils import plot_circle

'''
Plot relationship of the population between predator and prey
'''

argparser = argparse.ArgumentParser()
argparser.add_argument('--log_file', type=str, help='Path for a log file')
argparser.add_argument('--st', type=int, default=0, help='Start time step')
args = argparser.parse_args()

log_file = args.log_file

plot_circle(args.log_file, args.st)

