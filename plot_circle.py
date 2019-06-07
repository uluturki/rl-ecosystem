import os, sys

import matplotlib.pyplot as plt
import argparse
import numpy as np
import seaborn as sns
from utils import plot_circle

argparser = argparse.ArgumentParser()
argparser.add_argument('--log_file', type=str)
argparser.add_argument('--st', type=int, default=0)
args = argparser.parse_args()

log_file = args.log_file

plot_circle(args.log_file, args.st)

