# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:56:49 2014

@author: rlabbe
"""
import filterpy.stats as stats
from filterpy.stats import plot_covariance_ellipse
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import multivariate_normal

def show_residual_chart():
    est_y = ((164.2-158)*.8 + 158)

    ax = plt.axes(xticks=[], yticks=[], frameon=False)
    ax.annotate('', xy=[1,159], xytext=[0,158],
                arrowprops=dict(arrowstyle='->',
                                ec='r', lw=3, shrinkA=6, shrinkB=5))

    ax.annotate('', xy=[1,159], xytext=[1,164.2],
                arrowprops=dict(arrowstyle='-',
                                ec='k', lw=1, shrinkA=8, shrinkB=8))

    ax.annotate('', xy=(1., est_y), xytext=(0.9, est_y),
                arrowprops=dict(arrowstyle='->', ec='#004080',
                                lw=2,
                                shrinkA=3, shrinkB=4))


    plt.scatter ([0,1], [158.0,est_y], c='k',s=128)
    plt.scatter ([1], [164.2], c='b',s=128)
    plt.scatter ([1], [159], c='r', s=128)
    plt.text (1.0, 158.8, "prediction ($x_t)$", ha='center',va='top',fontsize=18,color='red')
    plt.text (1.0, 164.4, "measurement ($z$)",ha='center',va='bottom',fontsize=18,color='blue')
    plt.text (0, 157.8, "prior estimate ($\hat{x}_{t-1}$)", ha='center', va='top',fontsize=18)
    plt.text (1.02, est_y-1.5, "residual", ha='left', va='center',fontsize=18)
    plt.text (0.9, est_y, "new estimate ($\hat{x}_{t}$)", ha='right', va='center',fontsize=18)
    plt.xlabel('time')
    ax.yaxis.set_label_position("right")
    plt.ylabel('state')
    plt.xlim(-0.5, 1.5)
    plt.show()


def plot_gaussian_multiply():
    xs = np.arange(-5, 10, 0.1)

    mean1, var1 = 0, 5
    mean2, var2 = 5, 1
    mean, var = stats.mul(mean1, var1, mean2, var2)

    ys = [stats.gaussian(x, mean1, var1) for x in xs]
    plt.plot(xs, ys, label='M1')

    ys = [stats.gaussian(x, mean2, var2) for x in xs]
    plt.plot(xs, ys, label='M2')

    ys = [stats.gaussian(x, mean, var) for x in xs]
    plt.plot(xs, ys, label='M1 x M2')
    plt.legend()
    plt.show()


def show_position_chart():
    """ Displays 3 measurements at t=1,2,3, with x=1,2,3"""

    plt.scatter ([1,2,3], [1,2,3], s=128, color='#004080')
    plt.xlim([0,4]);
    plt.ylim([0,4])

    plt.annotate('t=1', xy=(1,1), xytext=(0,-10),
                  textcoords='offset points', ha='center', va='top')

    plt.annotate('t=2', xy=(2,2), xytext=(0,-10),
                  textcoords='offset points', ha='center', va='top')

    plt.annotate('t=3', xy=(3,3), xytext=(0,-10),
                  textcoords='offset points', ha='center', va='top')

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.xticks(np.arange(1,4,1))
    plt.yticks(np.arange(1,4,1))
    plt.show()


def show_position_prediction_chart():
    """ displays 3 measurements, with the next position predicted"""

    plt.scatter ([1,2,3], [1,2,3], s=128, color='#004080')

    plt.annotate('t=1', xy=(1,1), xytext=(0,-10),
                  textcoords='offset points', ha='center', va='top')

    plt.annotate('t=2', xy=(2,2), xytext=(0,-10),
                  textcoords='offset points', ha='center', va='top')

    plt.annotate('t=3', xy=(3,3), xytext=(0,-10),
                  textcoords='offset points', ha='center', va='top')

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


def show_x_error_chart(count):
    """ displays x=123 with covariances showing error"""

    plt.cla()
    plt.gca().autoscale(tight=True)

    cov = np.array([[0.03,0], [0,8]])
    e = stats.covariance_ellipse (cov)

    cov2 = np.array([[0.03,0], [0,4]])
    e2 = stats.covariance_ellipse (cov2)

    cov3 = np.array([[12,11.95], [11.95,12]])
    e3 = stats.covariance_ellipse (cov3)


    sigma=[1, 4, 9]

    if count >= 1:
        stats.plot_covariance_ellipse ((0,0), ellipse=e, variance=sigma)

    if count == 2 or count == 3:

        stats.plot_covariance_ellipse ((5,5), ellipse=e, variance=sigma)

    if count == 3:

        stats.plot_covariance_ellipse ((5,5), ellipse=e3, variance=sigma,
                                       edgecolor='r')

    if count == 4:
        M1 = np.array([[5, 5]]).T
        m4, cov4 = stats.multivariate_multiply(M1, cov2, M1, cov3)
        e4 = stats.covariance_ellipse (cov4)

        stats.plot_covariance_ellipse ((5,5), ellipse=e, variance=sigma,
                                       alpha=0.25)

        stats.plot_covariance_ellipse ((5,5), ellipse=e3, variance=sigma,
                                       edgecolor='r', alpha=0.25)
        stats.plot_covariance_ellipse (m4[:,0], ellipse=e4, variance=sigma)

    #plt.ylim([0,11])
    #plt.xticks(np.arange(1,4,1))

    plt.xlabel("Position")
    plt.ylabel("Velocity")

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


def plot_3_covariances():

    P = [[2, 0], [0, 2]]
    plt.subplot(131)
    plot_covariance_ellipse((2, 7), cov=P, facecolor='g', alpha=0.2,
                            title='|2 0|\n|0 2|', axis_equal=False)
    plt.ylim((4, 10))
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(132)
    P = [[2, 0], [0, 9]]
    plt.ylim((4, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    plot_covariance_ellipse((2, 7), P, facecolor='g', alpha=0.2,
                            axis_equal=False, title='|2 0|\n|0 9|')

    plt.subplot(133)
    P = [[2, 1.2], [1.2, 2]]
    plt.ylim((4, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    plot_covariance_ellipse((2, 7), P, facecolor='g', alpha=0.2,
                            axis_equal=False,
                            title='|2 1.2|\n|1.2 2|')

    plt.tight_layout()
    plt.show()


def plot_correlation_covariance():
    P = [[4, 3.9], [3.9, 4]]
    plot_covariance_ellipse((5, 10), P, edgecolor='k',
                            variance=[1, 2**2, 3**2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().autoscale(tight=True)
    plt.axvline(7.5, ls='--', lw=1)
    plt.axhline(12.5, ls='--', lw=1)
    plt.scatter(7.5, 12.5, s=2000, alpha=0.5)
    plt.title('|4.0 3.9|\n|3.9 4.0|')
    plt.show()


import book_plots as bp
def plot_track(ps, zs, cov,
               plot_P=True, y_lim=None,
               title='Kalman Filter'):

    count = len(zs)
    actual = np.linspace(0, count - 1, count)
    cov = np.asarray(cov)
    std = np.sqrt(cov[:,0,0])
    std_top = np.minimum(actual+std, [count + 10])
    std_btm = np.maximum(actual-std, [-50])

    std_top = actual+std
    std_btm = actual-std

    bp.plot_track(actual,c='k')
    bp.plot_measurements(range(1, count + 1), zs)
    bp.plot_filter(range(1, count + 1), ps)

    plt.plot(std_top, linestyle=':', color='k', lw=1, alpha=0.4)
    plt.plot(std_btm, linestyle=':', color='k', lw=1, alpha=0.4)
    plt.fill_between(range(len(std_top)), std_top, std_btm,
                     facecolor='yellow', alpha=0.2, interpolate=True)
    plt.legend(loc=4)
    if y_lim is not None:
        plt.ylim(y_lim)
    else:
        plt.ylim((-50, count + 10))

    plt.xlim((0,count))
    plt.title(title)
    plt.show()

    if plot_P:
        ax = plt.subplot(121)
        ax.set_title("$\sigma^2_x$")
        plot_covariance(cov, (0, 0))
        ax = plt.subplot(122)
        ax.set_title("$\sigma^2_y$")
        plot_covariance(cov, (1, 1))
        plt.show()

def plot_covariance(P, index=(0, 0)):
    ps = []
    for p in P:
        ps.append(p[index[0], index[1]])
    plt.plot(ps)



if __name__ == "__main__":
    pass
    #show_position_chart()
    #plot_3d_covariance((2,7), np.array([[8.,0],[0,4.]]))
    #plot_3d_sampled_covariance([2,7], [[8.,0],[0,4.]])
    #show_residual_chart()

    #show_position_chart()
    #show_x_error_chart(4)

