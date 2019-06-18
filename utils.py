import os, sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from cv2 import imread, VideoWriter
import cv2
import shutil

def plot_diversity(predators, preys, prefix_path, step):
    predator_liking_list = []
    predator_physical_list = []
    predator_environmental_condition_list = []
    prey_liking_list = []
    prey_physical_list = []
    prey_environmental_condition_list = []

    dir_name = os.path.join(prefix_path, 'diversity', str(step))

    try:
        os.makedirs(dir_name)
    except:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)

    for predator in predators:
        predator_liking_list.append(predator.liking)
        predator_physical_list.append(predator.physical)
        predator_environmental_condition_list.append(predator.environmental_condition)

    plt.figure()
    sns.set_style("darkgrid")
    plt.scatter(predator_liking_list, predator_physical_list)
    plt.xlabel('Liking')
    plt.ylabel('Physical')
    plt.show()
    plt.savefig(os.path.join(dir_name, 'predator_diversity_like_vs_physical.png'))

    plt.figure()
    sns.set_style("darkgrid")
    plt.scatter(predator_liking_list, predator_environmental_condition_list)
    plt.xlabel('Liking')
    plt.ylabel('Environmental Condition')
    plt.show()
    plt.savefig(os.path.join(dir_name, 'predator_diversity_like_vs_environmental.png'))

    sns.set_style("darkgrid")
    plt.figure()
    plt.scatter(predator_environmental_condition_list, predator_physical_list)
    plt.xlabel('Environmental Condition')
    plt.ylabel('Physical')
    plt.show()
    plt.savefig(os.path.join(dir_name, 'predator_diversity_environmental_vs_physical.png'))

    for prey in preys:
        prey_liking_list.append(prey.liking)
        prey_physical_list.append(prey.physical)
        prey_environmental_condition_list.append(prey.environmental_condition)

    plt.figure()
    sns.set_style("darkgrid")
    plt.scatter(prey_liking_list, prey_physical_list)
    plt.xlabel('Liking')
    plt.ylabel('Physical')
    plt.show()
    plt.savefig(os.path.join(dir_name, 'prey_diversity_like_vs_physical.png'))

    plt.figure()
    sns.set_style("darkgrid")
    plt.scatter(prey_liking_list, prey_environmental_condition_list)
    plt.xlabel('Liking')
    plt.ylabel('Environmental Condition')
    plt.show()
    plt.savefig(os.path.join(dir_name, 'prey_diversity_like_vs_environmental.png'))

    plt.figure()
    sns.set_style("darkgrid")
    plt.scatter(prey_environmental_condition_list, prey_physical_list)
    plt.xlabel('Environmental Condition')
    plt.ylabel('Physical')
    plt.show()
    plt.savefig(os.path.join(dir_name, 'prey_diversity_environmental_vs_physical.png'))


def plot_dynamics(log_file, st, mode='two_agents'):
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
            elif len(line) == 14:
                prey_num.append(int(line[9]))
                predator_num.append(int(line[11]))
    ed = len(predator_num)

    x = range(len(prey_num))
    plt.figure()
    sns.set_style("darkgrid")
    plt.plot(x[st:ed], predator_num[st:ed])
    plt.plot(x[st:ed], prey_num[st:ed])
    plt.legend(['Predators', 'Preys'])
    plt.xlabel('Timestep')
    plt.ylabel('Number of Agents')
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

    predator_num = predator_num[st:]
    prey_num = prey_num[st:]
    ed = len(predator_num)
    x = range(len(predator_num))
    length = len(predator_num)

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
    plt.xlabel('Number of predator agents')
    plt.ylabel('Number of prey agents')
 #   plt.legend(['predator number', 'prey number'], loc='upper left')
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(log_file),'circle.png'))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
