#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:28:01 2022

@author: florian
"""
import matplotlib.pyplot as plt
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from IPython import display
import os
from gym import wrappers

%matplotlib inline
path_project = os.path.abspath(os.path.join("__file__", ".."))
env = gym.make("ALE/SpaceInvaders-v5")

before_training = os.path.join(path_project, "before_training.mp4")
video = VideoRecorder(env, before_training)

for i in range(1000):
    
    #img=plt.imshow(env.render(mode='rgb_array'))
    #img.set_data(env.render(mode='rgb_array'))
    video.capture_frame()
    #display.display(plt.gcf())
    #display.clear_output(wait=True)
    env.step(env.action_space.sample()) # take a random action
video.close()
env.close()