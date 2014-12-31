# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:56:49 2014

@author: rlabbe
"""
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import multivariate_normal
import stats

def show_residual_chart():
    plt.xlim([0.9,2.5])
    plt.ylim([1.5,3.5])

    plt.scatter ([1,2,2],[2,3,2.3])
    plt.scatter ([2],[2.8],marker='o')
    ax = plt.axes()
    ax.annotate('', xy=(2,3), xytext=(1,2),
                arrowprops=dict(arrowstyle='->', ec='#004080',
                                lw=2,
                                shrinkA=3, shrinkB=4))
    ax.annotate('prediction', xy=(2.04,3.), color='#004080')
    ax.annotate('measurement', xy=(2.05, 2.28))
    ax.annotate('prior estimate', xy=(1, 1.9))
    ax.annotate('residual', xy=(2.04,2.6), color='#e24a33')
    ax.annotate('new estimate', xy=(2,2.8),xytext=(2.1,2.8),
                arrowprops=dict(arrowstyle='->', ec="k", shrinkA=3, shrinkB=4))
    ax.annotate('', xy=(2,3), xytext=(2,2.3),
                arrowprops=dict(arrowstyle="-",
                                ec="#e24a33",
                                lw=2,
                                shrinkA=5, shrinkB=5))
    plt.title("Kalman Filter Predict and Update")
    plt.axis('equal')
    plt.show()


def show_position_chart():
    """ Displays 3 measurements at t=1,2,3, with x=1,2,3"""

    plt.scatter ([1,2,3], [1,2,3], s=128, color='#004080')
    plt.xlim([0,4]);
    plt.ylim([0,4])

    plt.xlabel("Position")
    plt.ylabel("Time")

    plt.xticks(np.arange(1,4,1))
    plt.yticks(np.arange(1,4,1))
    plt.show()


def show_position_prediction_chart():
    """ displays 3 measurements, with the next position predicted"""

    plt.scatter ([1,2,3], [1,2,3], s=128, color='#004080')

    plt.xlim([0,5])
    plt.ylim([0,5])

    plt.xlabel("Position")
    plt.ylabel("Time")

    plt.xticks(np.arange(1,5,1))
    plt.yticks(np.arange(1,5,1))

    plt.scatter ([4], [4], c='g',s=128, color='#8EBA42')
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



def plot_3d_covariance(mean, cov):
    """ plots a 2x2 covariance matrix positioned at mean. mean will be plotted
    in x and y, and the probability in the z axis.

    Parameters
    ----------
    mean :  2x1 tuple-like object
        mean for x and y coordinates. For example (2.3, 7.5)

    cov : 2x2 nd.array
       the covariance matrix

    """

    # compute width and height of covariance ellipse so we can choose
    # appropriate ranges for x and y
    o,w,h = stats.covariance_ellipse(cov,3)
    # rotate width and height to x,y axis
    wx = abs(w*np.cos(o) + h*np.sin(o))*1.2
    wy = abs(h*np.cos(o) - w*np.sin(o))*1.2


    # ensure axis are of the same size so everything is plotted with the same
    # scale
    if wx > wy:
        w = wx
    else:
        w = wy

    minx = mean[0] - w
    maxx = mean[0] + w
    miny = mean[1] - w
    maxy = mean[1] + w

    xs = np.arange(minx, maxx, (maxx-minx)/40.)
    ys = np.arange(miny, maxy, (maxy-miny)/40.)
    xv, yv = np.meshgrid (xs, ys)

    zs = np.array([100.* stats.multivariate_gaussian(np.array([x,y]),mean,cov) \
                   for x,y in zip(np.ravel(xv), np.ravel(yv))])
    zv = zs.reshape(xv.shape)

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_surface(xv, yv, zv, rstride=1, cstride=1, cmap=cm.autumn)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.contour(xv, yv, zv, zdir='x', offset=minx-1, cmap=cm.autumn)
    ax.contour(xv, yv, zv, zdir='y', offset=maxy, cmap=cm.BuGn)


def plot_3d_sampled_covariance(mean, cov):
    """ plots a 2x2 covariance matrix positioned at mean. mean will be plotted
    in x and y, and the probability in the z axis.

    Parameters
    ----------
    mean :  2x1 tuple-like object
        mean for x and y coordinates. For example (2.3, 7.5)

    cov : 2x2 nd.array
       the covariance matrix

    """

    # compute width and height of covariance ellipse so we can choose
    # appropriate ranges for x and y
    o,w,h = stats.covariance_ellipse(cov,3)
    # rotate width and height to x,y axis
    wx = abs(w*np.cos(o) + h*np.sin(o))*1.2
    wy = abs(h*np.cos(o) - w*np.sin(o))*1.2


    # ensure axis are of the same size so everything is plotted with the same
    # scale
    if wx > wy:
        w = wx
    else:
        w = wy

    minx = mean[0] - w
    maxx = mean[0] + w
    miny = mean[1] - w
    maxy = mean[1] + w

    count = 1000
    x,y = multivariate_normal(mean=mean, cov=cov, size=count).T

    xs = np.arange(minx, maxx, (maxx-minx)/40.)
    ys = np.arange(miny, maxy, (maxy-miny)/40.)
    xv, yv = np.meshgrid (xs, ys)

    zs = np.array([100.* stats.multivariate_gaussian(np.array([xx,yy]),mean,cov) \
                   for xx,yy in zip(np.ravel(xv), np.ravel(yv))])
    zv = zs.reshape(xv.shape)

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(x,y, [0]*count, marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.contour(xv, yv, zv, zdir='x', offset=minx-1, cmap=cm.autumn)
    ax.contour(xv, yv, zv, zdir='y', offset=maxy, cmap=cm.BuGn)


if __name__ == "__main__":
    #show_position_chart()
    #plot_3d_covariance((2,7), np.array([[8.,0],[0,4.]]))
    plot_3d_sampled_covariance([2,7], [[8.,0],[0,4.]])
    #show_residual_chart()


