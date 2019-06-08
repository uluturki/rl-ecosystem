import os, sys

import numpy as np
import argparse
import cv2
from cv2 import imread, VideoWriter

argparser = argparse.ArgumentParser()

argparser.add_argument('--img_dir', type=str)

args = argparser.parse_args()

img_dir = args.img_dir

st = 0
ed = len(os.listdir(img_dir))

images = [os.path.join(img_dir, '{:d}.png'.format(i+1)) for i in range(st, ed)]

def make_video(images, outvid=None, fps=5, size=None, is_color=True, format="XVID"):
    """
    Create a video from a list of images.
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    """
    # fourcc = VideoWriter_fourcc(*format)
    # For opencv2 and opencv3:
    if int(cv2.__version__[0]) > 2:
        fourcc = cv2.VideoWriter_fourcc(*format)
    else:
        fourcc = cv2.cv.CV_FOURCC(*format)
    vid = None
    for image in images:
        assert os.path.exists(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()

make_video(images, os.path.join(img_dir, 'video.avi'))
