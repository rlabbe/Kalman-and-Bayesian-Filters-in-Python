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


from code.book_plots import figsize, end_interactive
from filterpy.monte_carlo import stratified_resample, residual_resample
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import randn, random, uniform, multivariate_normal, seed
#from nonlinear_plots import plot_monte_carlo_mean
import scipy.stats
#from RobotLocalizationParticleFilter import *



class ParticleFilter(object):

    def __init__(self, N, x_dim, y_dim):
        self.particles = np.empty((N, 3))  # x, y, heading
        self.N = N
        self.x_dim = x_dim
        self.y_dim = y_dim

        # distribute particles randomly with uniform weight
        self.weights = np.empty(N)
        self.weights.fill(1./N)
        self.particles[:, 0] = uniform(0, x_dim, size=N)
        self.particles[:, 1] = uniform(0, y_dim, size=N)
        self.particles[:, 2] = uniform(0, 2*np.pi, size=N)


    def predict(self, u, std):
        """ move according to control input u with noise std"""

        self.particles[:, 2] += u[0] + randn(self.N) * std[0]
        self.particles[:, 2] %= 2 * np.pi

        d = u[1] + randn(self.N)
        self.particles[:, 0] += np.cos(self.particles[:, 2]) * d
        self.particles[:, 1] += np.sin(self.particles[:, 2]) * d

        self.particles[:, 0:2] += u + randn(self.N, 2) * std


    def weight(self, z, var):
        dist = np.sqrt((self.particles[:, 0] - z[0])**2 +
                       (self.particles[:, 1] - z[1])**2)

        # simplification assumes variance is invariant to world projection
        n = scipy.stats.norm(0, np.sqrt(var))
        prob = n.pdf(dist)

        # particles far from a measurement will give us 0.0 for a probability
        # due to floating point limits. Once we hit zero we can never recover,
        # so add some small nonzero value to all points.
        prob += 1.e-12
        self.weights += prob
        self.weights /= sum(self.weights) # normalize


    def neff(self):
        return 1. / np.sum(np.square(self.weights))


    def resample(self):
        p = np.zeros((self.N, 3))
        w = np.zeros(self.N)

        cumsum = np.cumsum(self.weights)
        for i in range(self.N):
            index = np.searchsorted(cumsum, random())
            p[i] = self.particles[index]
            w[i] = self.weights[index]

        self.particles = p
        self.weights = w / np.sum(w)


    def estimate(self):
        """ returns mean and variance """
        pos = self.particles[:, 0:2]
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu)**2, weights=self.weights, axis=0)

        return mu, var


def plot_random_pd():
    def norm(x, x0, sigma):
        return np.exp(-0.5 * (x - x0) ** 2 / sigma ** 2)


    def sigmoid(x, x0, alpha):
        return 1. / (1. + np.exp(- (x - x0) / alpha))

    x = np.linspace(0, 1, 100)
    y2 =  (0.1 * np.sin(norm(x, 0.2, 0.05)) +  0.25 * norm(x, 0.6, 0.05) +
          .5*norm(x, .5, .08) +
           np.sqrt(norm(x, 0.8, 0.06)) +0.1 * (1 - sigmoid(x, 0.45, 0.15)))
    with plt.xkcd():
        #plt.setp(plt.gca().get_xticklabels(), visible=False)
        #plt.setp(plt.gca().get_yticklabels(), visible=False)
        plt.axes(xticks=[], yticks=[], frameon=False)
        plt.plot(x, y2)
        plt.ylim([0, max(y2)+.1])


def plot_monte_carlo_ukf():

    def f(x,y):
        return x+y, .1*x**2 + y*y

    mean = (0, 0)
    p = np.array([[32, 15], [15., 40.]])

    # Compute linearized mean
    mean_fx = f(*mean)

    #generate random points
    xs, ys = multivariate_normal(mean=mean, cov=p, size=3000).T
    fxs, fys = f(xs, ys)

    plt.subplot(121)
    plt.gca().grid(b=False)

    plt.scatter(xs, ys, marker='.', alpha=.2, color='k')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    plt.subplot(122)
    plt.gca().grid(b=False)

    plt.scatter(fxs, fys, marker='.', alpha=0.2, color='k')

    plt.ylim([-10, 200])
    plt.xlim([-100, 100])
    plt.show()


def plot_pf(pf, xlim=100, ylim=100, weights=True):

    if weights:
        a = plt.subplot(221)
        a.cla()

        plt.xlim(0, ylim)
        #plt.ylim(0, 1)
        a.set_yticklabels('')
        plt.scatter(pf.particles[:, 0], pf.weights, marker='.', s=1, color='k')
        a.set_ylim(bottom=0)

        a = plt.subplot(224)
        a.cla()
        a.set_xticklabels('')
        plt.scatter(pf.weights, pf.particles[:, 1], marker='.', s=1, color='k')
        plt.ylim(0, xlim)
        a.set_xlim(left=0)
        #plt.xlim(0, 1)

        a = plt.subplot(223)
        a.cla()

    else:
        plt.cla()
    plt.scatter(pf.particles[:, 0], pf.particles[:, 1], marker='.', s=1, color='k')
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)


def Gaussian(mu, sigma, x):

    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    g = (np.exp(-((mu - x) ** 2) / (sigma ** 2) / 2.0) /
         np.sqrt(2.0 * np.pi * (sigma ** 2)))
    for i in range(len(g)):
        g[i] = max(g[i], 1.e-29)

    return g


def test_gaussian(N):
    for i in range(N):
        mean, std, x = randn(3)
        std = abs(std)

        d = Gaussian(mean, std, x) - scipy.stats.norm(mean, std).pdf(x)
        assert abs(d) < 1.e-8, "{}, {}, {}, {}, {}, {}".format(d, mean, std, x, Gaussian(mean, std, x), scipy.stats.norm(mean, std).pdf(x))


def show_two_pf_plots():
    """ Displays results of PF after 1 and 10 iterations for the book.
    Note the book says this solves the full robot localization problem.
    It doesn't bother simulating landmarks as this is just an illustration.
    """

    seed(1234)
    N = 3000
    pf = ParticleFilter(N, 20, 20)
    z = np.array([20, 20])

    #plot(pf, weights=False)

    for x in range(10):

        z[0] = x+1 + randn()*0.3
        z[1] = x+1 + randn()*0.3

        pf.predict((1,1), (0.2, 0.2))
        pf.weight(z=z, var=.8)
        pf.resample()

        if x == 0:
            plt.subplot(121)
        elif x == 9:
            plt.subplot(122)

        if x == 0 or x == 9:
            mu, var = pf.estimate()
            plot_pf(pf, 20, 20, weights=False)
            if x == 0:
                plt.scatter(mu[0], mu[1], color='g', s=100)
                plt.scatter(x+1, x+1, marker='x', color='r', s=180, lw=3)
            else:
                plt.scatter(mu[0], mu[1], color='g', s=100, label="PF")
                plt.scatter([x+1], [x+1], marker='x', color='r', s=180, label="True", lw=3)
                plt.legend(scatterpoints=1)
            plt.tight_layout()


def test_pf():

    #seed(1234)
    N = 10000
    R = .2
    landmarks = [[-1, 2], [20,4], [10,30], [18,25]]
    #landmarks = [[-1, 2], [2,4]]

    pf = RobotLocalizationParticleFilter(N, 20, 20, landmarks, R)

    plot_pf(pf, 20, 20, weights=False)

    dt = .01
    plt.pause(dt)

    for x in range(18):

        zs = []
        pos=(x+3, x+3)

        for landmark in landmarks:
            d = np.sqrt((landmark[0]-pos[0])**2 +  (landmark[1]-pos[1])**2)
            zs.append(d + randn()*R)

        pf.predict((0.01, 1.414), (.2, .05))
        pf.update(z=zs)
        pf.resample()
        #print(x, np.array(list(zip(pf.particles, pf.weights))))

        mu, var = pf.estimate()
        plot_pf(pf, 20, 20, weights=False)
        plt.plot(pos[0], pos[1], marker='*', color='r', ms=10)
        plt.scatter(mu[0], mu[1], color='g', s=100)
        plt.tight_layout()
        plt.pause(dt)


def test_pf2():
    N = 1000
    sensor_std_err = .2
    landmarks = [[-1, 2], [20,4], [-20,6], [18,25]]

    pf = RobotLocalizationParticleFilter(N, 20, 20, landmarks, sensor_std_err)

    xs = []
    for x in range(18):
        zs = []
        pos=(x+1, x+1)

        for landmark in landmarks:
            d = np.sqrt((landmark[0]-pos[0])**2 +  (landmark[1]-pos[1])**2)
            zs.append(d + randn()*sensor_std_err)

        # move diagonally forward to (x+1, x+1)
        pf.predict((0.00, 1.414), (.2, .05))
        pf.update(z=zs)
        pf.resample()

        mu, var = pf.estimate()
        xs.append(mu)

    xs = np.array(xs)
    plt.plot(xs[:, 0], xs[:, 1])
    plt.show()



def plot_cumsum(a):

    with figsize(y=2):
        fig = plt.figure()
        N = len(a)

        cmap = mpl.colors.ListedColormap([[0., .4, 1.],
                                          [0., .8, 1.],
                                          [1., .8, 0.],
                                          [1., .4, 0.]]*(int(N/4) + 1))
        cumsum = np.cumsum(np.asarray(a) / np.sum(a))
        cumsum = np.insert(cumsum, 0, 0)

        #fig = plt.figure(figsize=(6,3))
        fig=plt.gcf()
        ax = fig.add_axes([0.05, 0.475, 0.9, 0.15])
        norm = mpl.colors.BoundaryNorm(cumsum, cmap.N)
        bar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                         norm=norm,
                                         drawedges=False,
                                         spacing='proportional',
                                         orientation='horizontal')
        if N > 10:
            bar.set_ticks([])
        end_interactive(fig)


def plot_stratified_resample(a):
    N = len(a)

    cmap = mpl.colors.ListedColormap([[0., .4, 1.],
                                      [0., .8, 1.],
                                      [1., .8, 0.],
                                      [1., .4, 0.]]*(int(N/4) + 1))
    cumsum = np.cumsum(np.asarray(a) / np.sum(a))
    cumsum = np.insert(cumsum, 0, 0)

    with figsize(y=2):
        fig = plt.figure()
        ax = plt.gcf().add_axes([0.05, 0.475, 0.9, 0.15])
        norm = mpl.colors.BoundaryNorm(cumsum, cmap.N)
        bar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                         norm=norm,
                                         drawedges=False,
                                         spacing='proportional',
                                         orientation='horizontal')
        xs = np.linspace(0., 1.-1./N, N)
        ax.vlines(xs, 0, 1, lw=2)

        # make N subdivisions, and chose a random position within each one
        b = (random(N) + range(N)) / N
        plt.scatter(b, [.5]*len(b), s=60, facecolor='k', edgecolor='k')
        bar.set_ticks([])
        plt.title('stratified resampling')
        end_interactive(fig)


def plot_systematic_resample(a):
    N = len(a)

    cmap = mpl.colors.ListedColormap([[0., .4, 1.],
                                      [0., .8, 1.],
                                      [1., .8, 0.],
                                      [1., .4, 0.]]*(int(N/4) + 1))
    cumsum = np.cumsum(np.asarray(a) / np.sum(a))
    cumsum = np.insert(cumsum, 0, 0)

    with figsize(y=2):
        fig = plt.figure()
        ax = plt.gcf().add_axes([0.05, 0.475, 0.9, 0.15])
        norm = mpl.colors.BoundaryNorm(cumsum, cmap.N)
        bar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                         norm=norm,
                                         drawedges=False,
                                         spacing='proportional',
                                         orientation='horizontal')
        xs = np.linspace(0., 1.-1./N, N)
        ax.vlines(xs, 0, 1, lw=2)

        # make N subdivisions, and chose a random position within each one
        b = (random() + np.array(range(N))) / N
        plt.scatter(b, [.5]*len(b), s=60, facecolor='k', edgecolor='k')
        bar.set_ticks([])
        plt.title('systematic resampling')
        end_interactive(fig)


def plot_multinomial_resample(a):
    N = len(a)

    cmap = mpl.colors.ListedColormap([[0., .4, 1.],
                                      [0., .8, 1.],
                                      [1., .8, 0.],
                                      [1., .4, 0.]]*(int(N/4) + 1))
    cumsum = np.cumsum(np.asarray(a) / np.sum(a))
    cumsum = np.insert(cumsum, 0, 0)

    with figsize(y=2):
        fig = plt.figure()
        ax = plt.gcf().add_axes([0.05, 0.475, 0.9, 0.15])
        norm = mpl.colors.BoundaryNorm(cumsum, cmap.N)
        bar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                         norm=norm,
                                         drawedges=False,
                                         spacing='proportional',
                                         orientation='horizontal')

        # make N subdivisions, and chose a random position within each one
        b = random(N)
        plt.scatter(b, [.5]*len(b), s=60, facecolor='k', edgecolor='k')
        bar.set_ticks([])
        plt.title('multinomial resampling')
        end_interactive(fig)


def plot_residual_resample(a):
    N = len(a)

    a_norm = np.asarray(a) / np.sum(a)
    cumsum = np.cumsum(a_norm)
    cumsum = np.insert(cumsum, 0, 0)

    cmap = mpl.colors.ListedColormap([[0., .4, 1.],
                                      [0., .8, 1.],
                                      [1., .8, 0.],
                                      [1., .4, 0.]]*(int(N/4) + 1))

    with figsize(y=2):
        fig = plt.figure()
        ax = plt.gcf().add_axes([0.05, 0.475, 0.9, 0.15])
        norm = mpl.colors.BoundaryNorm(cumsum, cmap.N)
        bar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                         norm=norm,
                                         drawedges=False,
                                         spacing='proportional',
                                         orientation='horizontal')

        indexes = residual_resample(a_norm)
        bins = np.bincount(indexes)
        for i in range(1, N):
            n =  bins[i-1] # number particles in this sample
            if n > 0:
                b = np.linspace(cumsum[i-1], cumsum[i], n+2)[1:-1]
                plt.scatter(b, [.5]*len(b), s=60, facecolor='k', edgecolor='k')
        bar.set_ticks([])
        plt.title('residual resampling')
        end_interactive(fig)


if __name__ == '__main__':
    plot_residual_resample([.1, .2, .3, .4, .2, .3, .1])

    #example()
    #show_two_pf_plots()

    a = [.1, .2, .1, .6]
    #plot_cumsum(a)
    #test_pf()
