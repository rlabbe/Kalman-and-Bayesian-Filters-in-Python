# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:21:40 2014

@author: rlabbe
"""
import matplotlib.pyplot as plt
import numpy as np

def plot(pos):
    plt.cla()
    ax = plt.gca()
    x = np.arange(len(pos))
    ax.bar(x, pos)
    plt.ylim([0,1])
    plt.xticks(x+0.5, x)
