# -*- coding: utf-8 -*-
"""
Created on Tue May 27 21:21:19 2014

@author: rlabbe
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Arrow
import stats
import numpy as np
import math
from filterpy.kalman import UnscentedKalmanFilter as UKF
from stats import plot_covariance_ellipse

def _sigma_points(mean, sigma, kappa):
    sigma1 = mean + math.sqrt((1+kappa)*sigma)
    sigma2 = mean - math.sqrt((1+kappa)*sigma)
    return mean, sigma1, sigma2


def arrow(x1,y1,x2,y2, width=0.2):
    return Arrow(x1,y1, x2-x1, y2-y1, lw=1, width=width, ec='k', color='k')


def show_two_sensor_bearing():
    circle1=plt.Circle((-4,0),5,color='#004080',fill=False,linewidth=20, alpha=.7)
    circle2=plt.Circle((4,0),5,color='#E24A33', fill=False, linewidth=5, alpha=.7)

    fig = plt.gcf()
    ax = fig.gca()

    plt.axis('equal')
    #plt.xlim((-10,10))
    plt.ylim((-6,6))

    plt.plot ([-4,0], [0,3], c='#004080')
    plt.plot ([4,0], [0,3], c='#E24A33')
    plt.text(-4, -.5, "A", fontsize=16, horizontalalignment='center')
    plt.text(4, -.5, "B", fontsize=16, horizontalalignment='center')

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    plt.show()


def show_three_gps():
    circle1=plt.Circle((-4,0),5,color='#004080',fill=False,linewidth=20, alpha=.7)
    circle2=plt.Circle((4,0),5,color='#E24A33', fill=False, linewidth=8, alpha=.7)
    circle3=plt.Circle((0,-3),6,color='#534543',fill=False, linewidth=13, alpha=.7)

    fig = plt.gcf()
    ax = fig.gca()

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    plt.axis('equal')
    plt.show()


def show_sigma_transform():
    fig = plt.figure()
    ax=fig.gca()

    x = np.array([0, 5])
    P = np.array([[4, -2.2], [-2.2, 3]])

    plot_covariance_ellipse(x, P, facecolor='b', variance=9, alpha=0.5)
    S = UKF.sigma_points(x=x, P=P, kappa=0)
    plt.scatter(S[:,0], S[:,1], c='k', s=80)

    x = np.array([15, 5])
    P = np.array([[3, 1.2],[1.2, 6]])
    plot_covariance_ellipse(x, P, facecolor='g', variance=9, alpha=0.5)


    ax.add_artist(arrow(S[0,0], S[0,1], 11, 4.1, 0.6))
    ax.add_artist(arrow(S[1,0], S[1,1], 13, 7.7, 0.6))
    ax.add_artist(arrow(S[2,0], S[2,1], 16.3, 0.93, 0.6))
    ax.add_artist(arrow(S[3,0], S[3,1], 16.7, 10.8, 0.6))
    ax.add_artist(arrow(S[4,0], S[4,1], 17.7, 5.6, 0.6))

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    #plt.axis('equal')
    plt.show()


def show_2d_transform():

    ax=plt.gca()

    ax.add_artist(Ellipse(xy=(2,5), width=2, height=3,angle=70,linewidth=1,ec='k'))
    ax.add_artist(Ellipse(xy=(7,5), width=2.2, alpha=0.3, height=3.8,angle=150,linewidth=1,ec='k'))

    ax.add_artist(arrow(2, 5, 6, 4.8))

    ax.add_artist(arrow(1.5, 5.5, 7, 3.8))
    ax.add_artist(arrow(2.3, 4.1, 8, 6))

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.axis('equal')
    plt.xlim(0,10); plt.ylim(0,10)
    plt.show()


def show_3_sigma_points():
    xs = np.arange(-4, 4, 0.1)
    var = 1.5
    ys = [stats.gaussian(x, 0, var) for x in xs]
    samples = [0, 1.2, -1.2]
    for x in samples:
        plt.scatter ([x], [stats.gaussian(x, 0, var)], s=80)

    plt.plot(xs, ys)
    plt.show()

def show_sigma_selections():
    ax=plt.gca()
    ax.add_artist(Ellipse(xy=(2,5), alpha=0.5, width=2, height=3,angle=0,linewidth=1,ec='k'))
    ax.add_artist(Ellipse(xy=(5,5), alpha=0.5, width=2, height=3,angle=0,linewidth=1,ec='k'))
    ax.add_artist(Ellipse(xy=(8,5), alpha=0.5, width=2, height=3,angle=0,linewidth=1,ec='k'))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.scatter([1.5,2,2.5],[5,5,5],c='k', s=50)
    plt.scatter([2,2],[4.5, 5.5],c='k', s=50)

    plt.scatter([4.8,5,5.2],[5,5,5],c='k', s=50)
    plt.scatter([5,5],[4.8, 5.2],c='k', s=50)

    plt.scatter([7.2,8,8.8],[5,5,5],c='k', s=50)
    plt.scatter([8,8],[4,6],c='k' ,s=50)

    plt.axis('equal')
    plt.xlim(0,10); plt.ylim(0,10)
    plt.show()



def show_sigmas_for_2_kappas():
    # generate the Gaussian data

    xs = np.arange(-4, 4, 0.1)
    mean = 0
    sigma = 1.5
    ys = [stats.gaussian(x, mean, sigma*sigma) for x in xs]



    #generate our samples
    kappa = 2
    x0,x1,x2 = _sigma_points(mean, sigma, kappa)

    samples = [x0,x1,x2]
    for x in samples:
        p1 = plt.scatter([x], [stats.gaussian(x, mean, sigma*sigma)], s=80, color='k')

    kappa = -.5
    x0,x1,x2 = _sigma_points(mean, sigma, kappa)

    samples = [x0,x1,x2]
    for x in samples:
        p2 = plt.scatter([x], [stats.gaussian(x, mean, sigma*sigma)], s=80, color='b')

    plt.legend([p1,p2], ['$kappa$=2', '$kappa$=-0.5'])
    plt.plot(xs, ys)
    plt.show()



if __name__ == '__main__':
    show_three_gps()
    #show_sigma_transform()
    #show_sigma_selections()

