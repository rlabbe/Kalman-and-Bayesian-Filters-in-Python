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
    plt.ylim([1.5,3.5])

    plt.scatter ([1,2,2],[2,3,2.3])
    plt.scatter ([2],[2.8],marker='o')
    ax = plt.axes()
    ax.annotate('', xy=(2,3), xytext=(1,2),
                arrowprops=dict(arrowstyle='->', ec='b',shrinkA=3, shrinkB=4))
    ax.annotate('prediction', xy=(2.04,3.), color='b')
    ax.annotate('measurement', xy=(2.05, 2.28))
    ax.annotate('prior estimate', xy=(1, 1.9))
    ax.annotate('residual', xy=(2.04,2.6), color='r')
    ax.annotate('new estimate', xy=(2,2.8),xytext=(2.1,2.8),
                arrowprops=dict(arrowstyle='->', ec="k", shrinkA=3, shrinkB=4))
    ax.annotate('', xy=(2,3), xytext=(2,2.3),
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
                arrowprops=dict(arrowstyle='->',
                                ec='g',
                                shrinkA=6, shrinkB=5,
                                lw=3))
    plt.show()


def show_x_error_chart():
    """ displays x=123 with covariances showing error"""

    cov = np.array([[0.003,0], [0,12]])
    sigma=[0.5,1.,1.5,2]
    e = stats.covariance_ellipse (cov)

    stats.plot_covariance_ellipse ((1,1), ellipse=e, variance=sigma, axis_equal=False)
    stats.plot_covariance_ellipse ((2,1), ellipse=e, variance=sigma, axis_equal=False)
    stats.plot_covariance_ellipse ((3,1), ellipse=e, variance=sigma, axis_equal=False)


    plt.ylim([0,11])
    plt.xticks(np.arange(1,4,1))

    plt.xlabel("Position")
    plt.ylabel("Time")

    plt.show()

def show_x_with_unobserved():
    """ shows x=1,2,3 with velocity superimposed on top """

    # plot velocity
    sigma=[0.5,1.,1.5,2]
    cov = np.array([[1,1],[1,1.1]])
    stats.plot_covariance_ellipse ((2,2), cov=cov, variance=sigma, axis_equal=False)

    # plot positions
    cov = np.array([[0.003,0], [0,12]])
    sigma=[0.5,1.,1.5,2]
    e = stats.covariance_ellipse (cov)

    stats.plot_covariance_ellipse ((1,1), ellipse=e, variance=sigma, axis_equal=False)
    stats.plot_covariance_ellipse ((2,1), ellipse=e, variance=sigma, axis_equal=False)
    stats.plot_covariance_ellipse ((3,1), ellipse=e, variance=sigma, axis_equal=False)

    # plot intersection cirle
    isct = Ellipse(xy=(2,2), width=.2, height=1.2, edgecolor='r', fc='None', lw=4)
    plt.gca().add_artist(isct)

    plt.ylim([0,11])
    plt.xlim([0,4])
    plt.xticks(np.arange(1,4,1))

    plt.xlabel("Position")
    plt.ylabel("Time")

    plt.show()



if __name__ == "__main__":

    show_residual_chart()
