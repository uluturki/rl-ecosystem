import os, sys

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_parameter, plot_parameter_var

argparser = argparse.ArgumentParser()
argparser.add_argument('--log_file', type=str)
argparser.add_argument('--st', type=int, default=0)
args = argparser.parse_args()

log_file = args.log_file

plot_parameter(args.log_file, args.st)


plot_parameter_var(args.log_file, args.st)