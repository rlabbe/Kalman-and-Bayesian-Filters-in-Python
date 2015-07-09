# -*- coding: utf-8 -*-
"""
Created on Tue May 27 21:21:19 2014

@author: rlabbe
"""
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import filterpy.stats as stats
from filterpy.stats import plot_covariance_ellipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Arrow
import math
import numpy as np

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
    fig = plt.figure()
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

def show_sigma_selections():
    ax=plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    x = np.array([2, 5])
    P = np.array([[3, 1.1], [1.1, 4]])

    points = MerweScaledSigmaPoints(2, .05, 2., 1.)
    sigmas = points.sigma_points(x, P)
    plot_covariance_ellipse(x, P, facecolor='b', alpha=0.6, variance=[.5])
    plt.scatter(sigmas[:,0], sigmas[:, 1], c='k', s=50)

    x = np.array([5, 5])
    points = MerweScaledSigmaPoints(2, .15, 2., 1.)
    sigmas = points.sigma_points(x, P)
    plot_covariance_ellipse(x, P, facecolor='b', alpha=0.6, variance=[.5])
    plt.scatter(sigmas[:,0], sigmas[:, 1], c='k', s=50)

    x = np.array([8, 5])
    points = MerweScaledSigmaPoints(2, .4, 2., 1.)
    sigmas = points.sigma_points(x, P)
    plot_covariance_ellipse(x, P, facecolor='b', alpha=0.6, variance=[.5])
    plt.scatter(sigmas[:,0], sigmas[:, 1], c='k', s=50)

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
    print(sum(Wc0))

def plot_radar(xs, t, plot_x=True, plot_vel=True, plot_alt=True):
    xs = np.asarray(xs)
    if plot_x:
        plt.figure()
        plt.plot(t, xs[:, 0]/1000.)
        plt.xlabel('time(sec)')
        plt.ylabel('position(km)')
    if plot_vel:
        plt.figure()
        plt.plot(t, xs[:, 1])
        plt.xlabel('time(sec)')
        plt.ylabel('velocity')
    if plot_alt:
        plt.figure()
        plt.plot(t, xs[:,2])
        plt.xlabel('time(sec)')
        plt.ylabel('altitude')
    plt.show()

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
    print('Difference in position in meters:', xs[-6:-1, 0] - Ms[-6:-1, 0])


if __name__ == '__main__':

    #show_2d_transform()
    #show_sigma_selections()

    show_sigma_transform(True)
    #show_four_gps()
    #show_sigma_transform()
    #show_sigma_selections()

