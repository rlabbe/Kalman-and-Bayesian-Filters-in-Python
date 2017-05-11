# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.


Code supporting the book

Kalman and Bayesian Filters in Python
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


This is licensed under an MIT license. See the LICENSE.txt file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import filterpy.stats as stats
from filterpy.stats import plot_covariance_ellipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Arrow
import math
from math import cos, sin, atan2, pi
import numpy as np
from numpy.random import randn

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


def show_four_gps():
    circle1=plt.Circle((-4,2),5,color='#004080',fill=False,linewidth=20, alpha=.7)
    circle2=plt.Circle((5.5,1),5,color='#E24A33', fill=False, linewidth=8, alpha=.7)
    circle3=plt.Circle((0,-3),6,color='#534543',fill=False, linewidth=13, alpha=.7)
    circle4=plt.Circle((0,8),5,color='#214513',fill=False, linewidth=13, alpha=.7)

    fig = plt.gcf()
    ax = fig.gca()

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)

    plt.axis('equal')
    plt.show()


def show_sigma_transform(with_text=False):
    fig = plt.gcf()
    ax=fig.gca()

    x = np.array([0, 5])
    P = np.array([[4, -2.2], [-2.2, 3]])

    plot_covariance_ellipse(x, P, facecolor='b', alpha=0.6, variance=9)
    sigmas = MerweScaledSigmaPoints(2, alpha=.5, beta=2., kappa=0.)

    S = sigmas.sigma_points(x=x, P=P)
    plt.scatter(S[:,0], S[:,1], c='k', s=80)

    x = np.array([15, 5])
    P = np.array([[3, 1.2],[1.2, 6]])
    plot_covariance_ellipse(x, P, facecolor='g', variance=9, alpha=0.3)

    ax.add_artist(arrow(S[0,0], S[0,1], 11, 4.1, 0.6))
    ax.add_artist(arrow(S[1,0], S[1,1], 13, 7.7, 0.6))
    ax.add_artist(arrow(S[2,0], S[2,1], 16.3, 0.93, 0.6))
    ax.add_artist(arrow(S[3,0], S[3,1], 16.7, 10.8, 0.6))
    ax.add_artist(arrow(S[4,0], S[4,1], 17.7, 5.6, 0.6))

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if with_text:
        plt.text(2.5, 1.5, r"$\chi$", fontsize=32)
        plt.text(13, -1, r"$\mathcal{Y}$", fontsize=32)

    #plt.axis('equal')
    plt.show()



def show_2d_transform():

    plt.cla()
    ax=plt.gca()

    ax.add_artist(Ellipse(xy=(2,5), width=2, height=3,angle=70,linewidth=1,ec='k'))
    ax.add_artist(Ellipse(xy=(7,5), width=2.2, alpha=0.3, height=3.8,angle=150,fc='g',linewidth=1,ec='k'))

    ax.add_artist(arrow(2, 5, 6, 4.8))
    ax.add_artist(arrow(1.5, 5.5, 7, 3.8))
    ax.add_artist(arrow(2.3, 4.1, 8, 6))
    ax.add_artist(arrow(3.3, 5.1, 6.5, 4.3))
    ax.add_artist(arrow(1.3, 4.8, 7.2, 6.3))
    ax.add_artist(arrow(1.1, 5.2, 8.2, 5.3))
    ax.add_artist(arrow(2, 4.4, 7.3, 4.5))

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

def _plot_sigmas(s, w, alpha=0.5, **kwargs):
    min_w = min(abs(w))
    scale_factor = 100 / min_w
    return plt.scatter(s[:, 0], s[:, 1], s=abs(w)*scale_factor,
                       alpha=alpha, **kwargs)


def show_sigma_selections():
    ax=plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    x = np.array([2, 5])
    P = np.array([[3, 1.1], [1.1, 4]])

    points = MerweScaledSigmaPoints(2, .09, 2., 1.)
    sigmas = points.sigma_points(x, P)
    Wm, Wc = points.weights()
    plot_covariance_ellipse(x, P, facecolor='b', alpha=.3, variance=[.5])
    _plot_sigmas(sigmas, Wc, alpha=1.0, facecolor='k')

    x = np.array([5, 5])
    points = MerweScaledSigmaPoints(2, .15, 1., .15)
    sigmas = points.sigma_points(x, P)
    Wm, Wc = points.weights()
    plot_covariance_ellipse(x, P, facecolor='b', alpha=0.3, variance=[.5])
    _plot_sigmas(sigmas, Wc, alpha=1.0, facecolor='k')

    x = np.array([8, 5])
    points = MerweScaledSigmaPoints(2, .2, 3., 10)
    sigmas = points.sigma_points(x, P)
    Wm, Wc = points.weights()
    plot_covariance_ellipse(x, P, facecolor='b', alpha=0.3, variance=[.5])
    _plot_sigmas(sigmas, Wc, alpha=1.0, facecolor='k')

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


def plot_sigma_points():
    x = np.array([0, 0])
    P = np.array([[4, 2], [2, 4]])

    sigmas = MerweScaledSigmaPoints(n=2, alpha=.3, beta=2., kappa=1.)
    S0 = sigmas.sigma_points(x, P)
    Wm0, Wc0 = sigmas.weights()

    sigmas = MerweScaledSigmaPoints(n=2, alpha=1., beta=2., kappa=1.)
    S1 = sigmas.sigma_points(x, P)
    Wm1, Wc1 = sigmas.weights()

    def plot_sigmas(s, w, **kwargs):
        min_w = min(abs(w))
        scale_factor = 100 / min_w
        return plt.scatter(s[:, 0], s[:, 1], s=abs(w)*scale_factor, alpha=.5, **kwargs)

    plt.subplot(121)
    plot_sigmas(S0, Wc0, c='b')
    plot_covariance_ellipse(x, P, facecolor='g', alpha=0.2, variance=[1, 4])
    plt.title('alpha=0.3')
    plt.subplot(122)
    plot_sigmas(S1, Wc1,  c='b', label='Kappa=2')
    plot_covariance_ellipse(x, P, facecolor='g', alpha=0.2, variance=[1, 4])
    plt.title('alpha=1')
    plt.show()


def plot_radar(xs, t, plot_x=True, plot_vel=True, plot_alt=True):
    xs = np.asarray(xs)
    if plot_x:
        plt.figure()
        plt.plot(t, xs[:, 0]/1000.)
        plt.xlabel('time(sec)')
        plt.ylabel('position(km)')
        plt.tight_layout()
    if plot_vel:
        plt.figure()
        plt.plot(t, xs[:, 1])
        plt.xlabel('time(sec)')
        plt.ylabel('velocity')
        plt.tight_layout()
    if plot_alt:
        plt.figure()
        plt.plot(t, xs[:,2])
        plt.xlabel('time(sec)')
        plt.ylabel('altitude')
        plt.tight_layout()
    plt.show()


def plot_altitude(xs, t, track):
    xs = np.asarray(xs)

    plt.plot(t, xs[:,2], label='filter', )
    plt.plot(t, track, label='Aircraft', lw=2, ls='--', c='k')
    plt.xlabel('time(sec)')
    plt.ylabel('altitude')
    plt.legend(loc=4)


def print_sigmas(n=1, mean=5, cov=3, alpha=.1, beta=2., kappa=2):
    points = MerweScaledSigmaPoints(n, alpha, beta, kappa)
    print('sigmas: ', points.sigma_points(mean,  cov).T[0])
    Wm, Wc = points.weights()
    print('mean weights:', Wm)
    print('cov weights:', Wc)
    print('lambda:', alpha**2 *(n+kappa) - n)
    print('sum cov', sum(Wc))




def plot_rts_output(xs, Ms, t):
    plt.figure()
    plt.plot(t, xs[:, 0]/1000., label='KF', lw=2)
    plt.plot(t, Ms[:, 0]/1000., c='k', label='RTS', lw=2)
    plt.xlabel('time(sec)')
    plt.ylabel('x')
    plt.legend(loc=4)

    plt.figure()

    plt.plot(t, xs[:, 1], label='KF')
    plt.plot(t, Ms[:, 1], c='k', label='RTS')
    plt.xlabel('time(sec)')
    plt.ylabel('x velocity')
    plt.legend(loc=4)

    plt.figure()
    plt.plot(t, xs[:, 2], label='KF')
    plt.plot(t, Ms[:, 2], c='k', label='RTS')
    plt.xlabel('time(sec)')
    plt.ylabel('Altitude(m)')
    plt.legend(loc=4)

    np.set_printoptions(precision=4)
    print('Difference in position in meters:\n\t', xs[-6:-1, 0] - Ms[-6:-1, 0])


def plot_scatter_of_bearing_error():
    def plot_scatter(theta):
        theta = math.radians(theta)
        d = 100
        xs, ys = [], []
        for i in range (3000):
            a = theta + randn() * math.radians(1)
            xs.append(d*math.cos(a))
            ys.append(d*math.sin(a))
        plt.scatter(xs, ys)
        plt.xlabel('x')
        plt.ylabel('y')

    plt.subplot(121)
    plot_scatter(45)
    plt.gca().set_aspect('equal')
    plt.title("45° bearing")
    plt.subplot(122)
    plot_scatter(180)
    plt.xlim((-101, -99))
    plt.title("180° bearing")


def plot_scatter_moving_target():
    pos = np.array([5., 5.])
    for i in range(5):
        pos += (0.5, 1.)
        actual_angle = math.atan2(pos[1], pos[0])
        d = math.sqrt(pos[0]**2 + pos[1]**2)

        xs, ys = [], []
        for i in range (100):
            a = actual_angle + randn() * math.radians(1)
            xs.append(d*math.cos(a))
            ys.append(d*math.sin(a))
        plt.scatter(xs, ys)

    plt.axis('equal')
    plt.plot([5.5, pos[0]], [6, pos[1]], c='g', linestyle='--')


def _isct(pa, pb, alpha, beta):
    """ Returns the (x, y) intersections of points pa and pb
    given the bearing ba for point pa and bearing bb for
    point pb.
    """

    B = [pb[0] - pa[0], pb[1] - pa[1]]
    AB = math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
    ab = atan2(B[1], B[0])
    a = alpha - ab
    b = pi - beta - ab
    p = pi - b - a

    AP = (sin(b) / sin(p)) * AB
    x = cos(alpha) * AP + pa[0]
    y = sin(alpha) * AP + pa[1]
    return x, y


def _plot_iscts(pos, sa, sb, N=4):
    for i in range(N):
        pos += (0.5, 1.)
        actual_angle_a = math.atan2(pos[1] - sa[1], pos[0] - sa[0])
        actual_angle_b = math.atan2(pos[1] - sb[1], pos[0] - sb[0])

        da = math.sqrt((sa[0] - pos[0])**2 + (sa[1] - pos[1])**2)
        db = math.sqrt((sb[0] - pos[0])**2 + (sb[1] - pos[1])**2)

        xs, ys, xs_a, xs_b, ys_a, ys_b = [], [], [], [], [], []

        for i in range (300):
            a_a = actual_angle_a + randn() * math.radians(1)
            a_b = actual_angle_b + randn() * math.radians(1)

            x,y = _isct(sa, sb, a_a, a_b)
            xs.append(x)
            ys.append(y)

            xs_a.append(da*math.cos(a_a) + sa[0])
            ys_a.append(da*math.sin(a_a) + sa[1])

            xs_b.append(db*math.cos(a_b) + sb[0])
            ys_b.append(db*math.sin(a_b) + sb[1])

        plt.scatter(xs, ys, c='r', marker='.', alpha=0.5)
        plt.scatter(xs_a, ys_a, c='k', edgecolor='k')
        plt.scatter(xs_b, ys_b, marker='v', edgecolor=None)
    plt.gca().set_aspect('equal')


def plot_iscts_two_sensors():
    plt.subplot(121)
    pos = np.array([4., 4,])
    sa = [0., 2.]
    sb = [8., 2.]

    plt.scatter(*sa, s=200, c='k', marker='v')
    plt.scatter(*sb, s=200, marker='s')
    _plot_iscts(pos, sa, sb, N=4)
    plt.subplot(122)
    plot_iscts_two_sensors_changed_sensors()



def plot_iscts_two_sensors_changed_sensors():
    sa = [3, 4]
    sb = [3, 7]
    pos= np.array([3., 3.])

    plt.scatter(*sa, s=200, c='k', marker='v')
    plt.scatter(*sb, s=200, marker='s')
    _plot_iscts(pos, sa, sb, N=5)
    plt.ylim(3.8, 8.5)


if __name__ == '__main__':

    #show_2d_transform()
    #show_sigma_selections()
    plot_scatter_of_bearing_error()

    #show_sigma_transform(True)
    #show_four_gps()
    #show_sigma_transform()
    #show_sigma_selections()

