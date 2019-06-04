import os, sys

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import spline
from utils import plot_dynamics

argparser = argparse.ArgumentParser()
argparser.add_argument('--log_file', type=str)
argparser.add_argument('--st', type=int, default=0)
args = argparser.parse_args()

log_file = args.log_file

plot_dynamics(args.log_file, args.st)
