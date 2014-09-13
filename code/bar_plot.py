# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:21:40 2014

@author: rlabbe
"""
import matplotlib.pyplot as plt
import numpy as np

def plot(pos, ylim=(0,1), title=None):
    plt.cla()
    ax = plt.gca()
    x = np.arange(len(pos))
    ax.bar(x, pos)
    if ylim:
        plt.ylim(ylim)
    plt.xticks(x+0.4, x)
    plt.grid()
    if title is not None:
        plt.title(title)


if __name__ == "__main__":
    p = [0.2245871, 0.06288015, 0.06109133, 0.0581008, 0.09334062, 0.2245871,
     0.06288015, 0.06109133, 0.0581008,  0.09334062]*2
    plot(p)