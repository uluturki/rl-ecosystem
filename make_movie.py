import os, sys

import numpy as np
import argparse
import cv2
from cv2 import imread, VideoWriter
from utils import make_video

argparser = argparse.ArgumentParser()

argparser.add_argument('--img_dir', type=str)

args = argparser.parse_args()

img_dir = args.img_dir

st = 0
ed = len(os.listdir(img_dir))

images = [os.path.join(img_dir, '{:d}.png'.format(i+1)) for i in range(st, ed)]

make_video(images, os.path.join(img_dir, 'video.avi'))
