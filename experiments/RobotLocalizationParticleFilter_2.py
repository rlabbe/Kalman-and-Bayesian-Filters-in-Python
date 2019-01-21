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

import numpy as np

from numpy.random import randn, random, uniform
import scipy.stats



def create_uniform_particles( x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi

    return particles


def create_gaussian_particles( mean, var, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + randn(N)*var[0]
    particles[:, 1] = mean[1] + randn(N)*var[1]
    particles[:, 2] = mean[2] + randn(N)*var[2]
    particles[:, 2] %= 2 * np.pi
    return particles



def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise `std (std_heading, std`"""

    N = len(particles)

    particles[:, 2] += u[0] + randn(N) * std[0]
    particles[:, 2] %= 2 * np.pi

    d = u[1]*dt + randn(N) * std[1]
    particles[:, 0] += np.cos(particles[:, 2]) * d
    particles[:, 1] += np.sin(particles[:, 2]) * d


def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300
    weights /= sum(weights) # normalize


def neff(weights):
    return 1. / np.sum(np.square(weights))


def resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights) # normalize


def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)


def estimate(particles, weights):
    """ returns mean and variance """
    pos = particles[:, 0:2]
    mu = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mu)**2, weights=weights, axis=0)

    return mu, var


def mean(particles, weights):
    """ returns weighted mean position"""
    return np.average(particles[:, 0:2], weights=weights, axis=0)



def residual_resample(w):

    N = len(w)

    w_ints = np.floor(N*w).astype(int)
    residual = w - w_ints
    residual /= sum(residual)

    indexes = np.zeros(N, 'i')
    k = 0
    for i in range(N):
        for j in range(w_ints[i]):
            indexes[k] = i
            k += 1
    cumsum = np.cumsum(residual)
    cumsum[N-1] = 1.
    for j in range(k, N):
        indexes[j] = np.searchsorted(cumsum, random())

    return indexes



def residual_resample2(w):

    N = len(w)

    w_ints =np.floor(N*w).astype(int)

    R = np.sum(w_ints)
    m_rdn = N - R

    Ws = (N*w - w_ints)/ m_rdn
    indexes = np.zeros(N, 'i')
    i = 0
    for j in range(N):
        for k in range(w_ints[j]):
            indexes[i] = j
            i += 1
    cumsum = np.cumsum(Ws)
    cumsum[N-1] = 1 # just in case

    for j in range(i, N):
        indexes[j] = np.searchsorted(cumsum, random())

    return indexes



def systemic_resample(w):
    N = len(w)
    Q = np.cumsum(w)
    indexes = np.zeros(N, 'int')
    t = np.linspace(0, 1-1/N, N) + random()/N

    i, j = 0, 0
    while i < N and j < N:
        while Q[j] < t[i]:
            j += 1
        indexes[i] = j
        i += 1

    return indexes





def Gaussian(mu, sigma, x):

    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    g = (np.exp(-((mu - x) ** 2) / (sigma ** 2) / 2.0) /
         np.sqrt(2.0 * np.pi * (sigma ** 2)))
    for i in range(len(g)):
        g[i] = max(g[i], 1.e-229)
    return g


if __name__ == '__main__':

    DO_PLOT_PARTICLES = False
    from numpy.random import seed
    import matplotlib.pyplot as plt

    #plt.figure()

    seed(5)
    for count in range(10):
        print()
        print(count)

        N = 4000
        sensor_std_err = .1
        landmarks = np.array([[-1, 2], [2,4], [10,6], [18,25]])
        NL = len(landmarks)


        particles = create_uniform_particles((0,20), (0,20), (0, 6.28), N)
        weights = np.zeros(N)

        #if DO_PLOT_PARTICLES:
        #    plt.scatter(particles[:, 0], particles[:, 1], alpha=.2, color='g')

        xs = []
        for x in range(18):
            zs = []
            pos=(x+1, x+1)

            for landmark in landmarks:
                d = np.sqrt((landmark[0]-pos[0])**2 +  (landmark[1]-pos[1])**2)
                zs.append(d + randn()*sensor_std_err)


            zs = np.linalg.norm(landmarks - pos, axis=1) + randn(NL)*sensor_std_err


            # move diagonally forward to (x+1, x+1)

            predict(particles, (0.00, 1.414), (.2, .05))

            update(particles, weights, z=zs, R=sensor_std_err, landmarks=landmarks)
            if x == 0:
                print(max(weights))
            #while abs(pf.neff() -N) < .1:
            #    print('neffing')
            #    pf.create_uniform_particles((0,20), (0,20), (0, 6.28))
            #    pf.update(z=zs)
            #print(pf.neff())
            #indexes = residual_resample2(pf.weights)
            indexes = systemic_resample(weights)

            resample_from_index(particles, weights, indexes)
            #pf.resample()

            mu, var = estimate(particles, weights)
            xs.append(mu)
            if DO_PLOT_PARTICLES:
                plt.scatter(particles[:, 0], particles[:, 1], alpha=.2)
                plt.scatter(pos[0], pos[1], marker='*', color='r')
                plt.scatter(mu[0], mu[1], marker='s', color='r')
                plt.pause(.01)

        xs = np.array(xs)
        plt.plot(xs[:, 0], xs[:, 1])
        plt.show()
