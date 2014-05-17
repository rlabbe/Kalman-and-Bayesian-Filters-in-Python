# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:56:49 2014

@author: rlabbe
"""
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import stats

def show_residual_chart():
    plt.xlim([0.9,2.5])
    plt.ylim([0.5,2.5])

    plt.scatter ([1,2,2],[1,2,1.3])
    plt.scatter ([2],[1.8],marker='o')
    ax = plt.axes()
    ax.annotate('', xy=(2,2), xytext=(1,1),
                arrowprops=dict(arrowstyle='->', ec='b',shrinkA=3, shrinkB=4))
    ax.annotate('prediction', xy=(2.04,2.), color='b')
    ax.annotate('measurement', xy=(2.05, 1.28))
    ax.annotate('prior measurement', xy=(1, 0.9))
    ax.annotate('residual', xy=(2.04,1.6), color='r')
    ax.annotate('new estimate', xy=(2,1.8),xytext=(2.1,1.8),
                arrowprops=dict(arrowstyle='->', ec="k", shrinkA=3, shrinkB=4))
    ax.annotate('', xy=(2,2), xytext=(2,1.3),
                arrowprops=dict(arrowstyle="-",
                                ec="r",
                                shrinkA=5, shrinkB=5))
    plt.title("Kalman Filter Prediction Update Step")
    plt.show()


def show_position_chart():
    """ Displays 3 measurements at t=1,2,3, with x=1,2,3"""

    plt.scatter ([1,2,3], [1,2,3], s=128)
    plt.xlim([0,4]);
    plt.ylim([0,4])

    plt.xlabel("Position")
    plt.ylabel("Time")

    plt.xticks(np.arange(1,4,1))
    plt.yticks(np.arange(1,4,1))
    plt.show()

def show_position_prediction_chart():
    """ displays 3 measurements, with the next position predicted"""

    plt.scatter ([1,2,3], [1,2,3], s=128)

    plt.xlim([0,5])
    plt.ylim([0,5])

    plt.xlabel("Position")
    plt.ylabel("Time")

    plt.xticks(np.arange(1,5,1))
    plt.yticks(np.arange(1,5,1))

    plt.scatter ([4], [4], c='g',s=128)
    ax = plt.axes()
    ax.annotate('', xy=(4,4), xytext=(3,3),
                arrowprops=dict(arrowstyle='->', ec='g',shrinkA=6, lw=3,shrinkB=5))
    plt.show()


def show_x_error_char():
    """ displays x=123 with covariances showing error"""

    cov = np.array([[0.003,0], [0,12]])
    sigma=[0.5,1.,1.5,2]

    e1 = stats.sigma_ellipses(cov, x=1, y=1, sigma=sigma)
    e2 = stats.sigma_ellipses(cov, x=2, y=2, sigma=sigma)
    e3 = stats.sigma_ellipses(cov, x=3, y=3, sigma=sigma)

    stats.plot_sigma_ellipses([e1, e2, e3], axis_equal=True,x_lim=[0,4],y_lim=[0,15])

    plt.ylim([0,11])
    plt.xticks(np.arange(1,4,1))

    plt.xlabel("Position")
    plt.ylabel("Time")

    plt.show()

def show_x_with_unobserved():
    """ shows x=1,2,3 with velocity superimposed on top """

    sigma=[0.5,1.,1.5,2]
    cov = np.array([[1,1],[1,1.1]])
    ev = stats.sigma_ellipses(cov, x=2, y=2, sigma=sigma)

    cov = np.array([[0.003,0], [0,12]])
    e1 = stats.sigma_ellipses(cov, x=1, y=1, sigma=sigma)
    e2 = stats.sigma_ellipses(cov, x=2, y=2, sigma=sigma)
    e3 = stats.sigma_ellipses(cov, x=3, y=3, sigma=sigma)

    isct = Ellipse(xy=(2,2), width=.2, height=1.2, edgecolor='r', fc='None', lw=4)
    plt.figure().gca().add_artist(isct)
    stats.plot_sigma_ellipses([e1, e2, e3, ev], axis_equal=True,x_lim=[0,4],y_lim=[0,15])

    plt.ylim([0,11])
    plt.xticks(np.arange(1,4,1))

    plt.xlabel("Position")
    plt.ylabel("Time")

    plt.show()