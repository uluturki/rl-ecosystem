import os, sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from cv2 import imread, VideoWriter, resize
import cv2
import shutil

def plot_cumulative_variation(log_file, st=0, ed=None):
    random_predators = []
    trained_predators = []
    training_predators = []

    random_preys = []
    trained_preys = []
    training_preys = []

    with open(log_file) as f:
        for line in f:
            line = line.split()
            random_preys.append(int(line[7]))
            trained_preys.append(int(line[9]))
            training_preys.append(int(line[11]))
            random_predators.append(int(line[13]))
            trained_predators.append(int(line[15]))
            training_predators.append(int(line[17]))

    if ed is None:
        ed = len(random_predators)

    random_predators = np.array(random_predators)[st:ed]
    trained_predators = np.array(trained_predators)[st:ed]
    training_predators = np.array(training_predators)[st:ed]
    random_preys = np.array(random_preys)[st:ed]
    trained_preys = np.array(trained_preys)[st:ed]
    training_preys = np.array(training_preys)[st:ed]

    x = range(len(random_preys))
    sns.set_style("darkgrid")
    fig, axs = plt.subplots(3, figsize=(22, 18))
    axs[0].fill_between(x, np.zeros(len(random_preys)), random_preys, alpha=0.3)
    axs[0].fill_between(x, random_preys, random_preys+trained_preys, alpha=0.3)
    axs[0].fill_between(x, random_preys+trained_preys, random_preys+trained_preys+training_preys, alpha=0.3)
    axs[0].legend(['Random Preys', 'Trained Preys', 'Training Preys'])
    axs[0].set_xlabel('Timestep')
    axs[0].set_ylabel('Number of Agents')
    axs[1].fill_between(x, np.zeros(len(random_predators)), random_predators, alpha=0.3)
    axs[1].fill_between(x, random_predators, random_predators+trained_predators, alpha=0.3)
    axs[1].fill_between(x, random_predators+trained_predators, random_predators+trained_predators+training_predators, alpha=0.3)
    axs[1].legend(['Random Predators', 'Trained Predators', 'Training Predators'])
    axs[1].set_xlabel('Timestep')
    axs[1].set_ylabel('Number of Agents')

    sns.set_style("darkgrid")
    axs[2].plot(x, random_predators+trained_predators+training_predators)
    axs[2].plot(x, random_preys+trained_preys+training_preys)
    axs[2].legend(['Predators', 'Preys'])
    axs[2].set_xlabel('Timestep')
    axs[2].set_ylabel('Number of Agents')
    plt.show()

    plt.savefig(os.path.join(os.path.dirname(log_file),'variation.png'))

def plot_parameter_var(log_file, st=0):
    prey = {'health': [], 'attack': [], 'resilience': [], 'speed': []}
    predator = {'health': [], 'attack': [], 'resilience': [], 'speed': []}
    with open(log_file)as fin:
        for line in fin:
            line = line.split()
            #if len(line) == 12:
            #    prey_num.append(int(line[9]))
            #    predator_num.append(int(line[11]))
            #elif len(line) == 10:
            predator['health'].append(float(line[33]))
            prey['health'].append(float(line[35]))
            predator['attack'].append(float(line[37]))
            prey['attack'].append(float(line[39]))
            predator['resilience'].append(float(line[41]))
            prey['resilience'].append(float(line[43]))
            predator['speed'].append(float(line[45]))
            prey['speed'].append(float(line[47]))

            #elif len(line) == 14:
            #    prey_num.append(int(line[9]))
            #    predator_num.append(int(line[11]))
    ed = len(prey['health'])

    x = range(st, ed)
    plt.figure()
    sns.set_style("darkgrid")
  #  plt.plot(x, predator['health'][st:ed])
  #  plt.plot(x, prey['health'][st:ed])
    plt.plot(x, predator['attack'][st:ed])
    plt.plot(x, prey['attack'][st:ed])
    plt.plot(x, predator['resilience'][st:ed])
    plt.plot(x, prey['resilience'][st:ed])
    plt.plot(x, predator['speed'][st:ed])
    plt.plot(x, prey['speed'][st:ed])
    plt.legend(['Predator Attack', 'Prey Attack', 'Predator Resilience', 'Prey Resilience', 'Predator Speed', 'Prey Speed'])
    plt.xlabel('Timestep')
    plt.ylabel('Values')
    plt.show()
    plt.savefig(os.path.join(os.path.dirname(log_file),'agent_parameter_variance.png'))

def plot_parameter(log_file, st=0):
    prey = {'health': [], 'attack': [], 'resilience': [], 'speed': []}
    predator = {'health': [], 'attack': [], 'resilience': [], 'speed': []}
    with open(log_file)as fin:
        for line in fin:
            line = line.split()
            #if len(line) == 12:
            #    prey_num.append(int(line[9]))
            #    predator_num.append(int(line[11]))
            #elif len(line) == 10:
            predator['health'].append(float(line[17]))
            prey['health'].append(float(line[19]))
            predator['attack'].append(float(line[21]))
            prey['attack'].append(float(line[23]))
            predator['resilience'].append(float(line[25]))
            prey['resilience'].append(float(line[27]))
            if len(line) > 28:
                predator['speed'].append(float(line[29]))
                prey['speed'].append(float(line[31]))

            #elif len(line) == 14:
            #    prey_num.append(int(line[9]))
            #    predator_num.append(int(line[11]))
    ed = len(prey['health'])

    x = range(st, ed)
    plt.figure()
    sns.set_style("darkgrid")
  #  plt.plot(x, predator['health'][st:ed])
  #  plt.plot(x, prey['health'][st:ed])
    plt.plot(x, predator['attack'][st:ed])
    plt.plot(x, prey['attack'][st:ed])
    plt.plot(x, predator['resilience'][st:ed])
    plt.plot(x, prey['resilience'][st:ed])
    if len(line) > 28:
        plt.plot(x, predator['speed'][st:ed])
        plt.plot(x, prey['speed'][st:ed])
        plt.legend(['Predator Attack', 'Prey Attack', 'Predator Resilience', 'Prey Resilience', 'Predator Speed', 'Prey Speed'])
        #plt.legend(['Predator Health', 'Prey Health', 'Predator Attack', 'Prey Attack', 'Predator Resilience', 'Prey Resilience', 'Predator Speed', 'Prey Speed'])
    else:
        #plt.legend(['Predator Health', 'Prey Health', 'Predator Attack', 'Prey Attack', 'Predator Resilience', 'Prey Resilience'])
        plt.legend(['Predator Attack', 'Prey Attack', 'Predator Resilience', 'Prey Resilience'])
    plt.xlabel('Timestep')
    plt.ylabel('Values')
    plt.show()
    plt.savefig(os.path.join(os.path.dirname(log_file),'agent_parameter.png'))


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


def plot_dynamics(log_file, st):
    prey_num = []
    predator_num = []
    with open(log_file)as fin:
        for line in fin:
            line = line.split()
            #if len(line) == 12:
            #    prey_num.append(int(line[9]))
            #    predator_num.append(int(line[11]))
            #elif len(line) == 10:
            prey_num.append(int(line[7]))
            predator_num.append(int(line[9]))
            #elif len(line) == 14:
            #    prey_num.append(int(line[9]))
            #    predator_num.append(int(line[11]))
    ed = len(predator_num)

    x = range(len(prey_num))
    plt.figure(figsize=(18, 6))
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
    sns.set_style("darkgrid")
    plt.grid()
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
