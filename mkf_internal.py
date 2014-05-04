# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:56:49 2014

@author: rlabbe
"""
import matplotlib.pyplot as plt

def show_residual_chart():
    plt.xlim([0.9,2.5])
    plt.ylim([0.5,2.5])

    plt.scatter ([1,2,2],[1,2,1.3])
    plt.scatter ([2],[1.8],marker='o')
    ax = plt.axes()
    ax.annotate('', xy=(2,2), xytext=(1,1),
                arrowprops=dict(arrowstyle='->', ec='b',shrinkA=3, shrinkB=4))
    ax.annotate('prediction', xy=(1.7,2), color='b')
    ax.annotate('measurement', xy=(2.05, 1.28))
    ax.annotate('prior measurement', xy=(1, 0.9))
    ax.annotate('residual', xy=(2.04,1.6), color='r')
    ax.annotate('new estimate', xy=(2,1.8),xytext=(2.15,1.9),
                arrowprops=dict(arrowstyle='->', shrinkA=3, shrinkB=4))
    ax.annotate('', xy=(2,2), xytext=(2,1.3),
                arrowprops=dict(arrowstyle="<->",
                                ec="r",
                                shrinkA=5, shrinkB=5))
    plt.title("Kalman Filter Prediction Update Step")
    plt.show()