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


class RobotLocalizationParticleFilter(object):

    def __init__(self, N, x_dim, y_dim, landmarks, measure_std_error):
        self.particles = np.empty((N, 3))  # x, y, heading
        self.N = N
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.landmarks = landmarks
        self.R = measure_std_error

        # distribute particles randomly with uniform weight
        self.weights = np.empty(N)
        #self.weights.fill(1./N)
        '''self.particles[:, 0] = uniform(0, x_dim, size=N)
        self.particles[:, 1] = uniform(0, y_dim, size=N)
        self.particles[:, 2] = uniform(0, 2*np.pi, size=N)'''


    def create_uniform_particles(self, x_range, y_range, hdg_range):
        self.particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
        self.particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
        self.particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
        self.particles[:, 2] %= 2 * np.pi

    def create_gaussian_particles(self, mean, var):
        self.particles[:, 0] = mean[0] + randn(self.N)*var[0]
        self.particles[:, 1] = mean[1] + randn(self.N)*var[1]
        self.particles[:, 2] = mean[2] + randn(self.N)*var[2]
        self.particles[:, 2] %= 2 * np.pi


    def predict(self, u, std, dt=1.):
        """ move according to control input u (heading change, velocity)
        with noise std"""

        self.particles[:, 2] += u[0] + randn(self.N) * std[0]
        self.particles[:, 2] %= 2 * np.pi

        d = u[1]*dt + randn(self.N) * std[1]
        self.particles[:, 0] += np.cos(self.particles[:, 2]) * d
        self.particles[:, 1] += np.sin(self.particles[:, 2]) * d


    def update(self, z):
        self.weights.fill(1.)
        for i, landmark in enumerate(self.landmarks):
            distance = np.linalg.norm(self.particles[:, 0:2] - landmark, axis=1)
            self.weights *= scipy.stats.norm(distance, self.R).pdf(z[i])
            #self.weights *= Gaussian(distance, self.R, z[i])

        self.weights += 1.e-300
        self.weights /= sum(self.weights) # normalize


    def neff(self):
        return 1. / np.sum(np.square(self.weights))


    def resample(self):
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, random(self.N))

        # resample according to indexes
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights /= np.sum(self.weights) # normalize


    def resample_from_index(self, indexes):
        assert len(indexes) == self.N

        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights /= np.sum(self.weights)


    def estimate(self):
        """ returns mean and variance """
        pos = self.particles[:, 0:2]
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu)**2, weights=self.weights, axis=0)

        return mu, var

    def mean(self):
        """ returns weighted mean position"""
        return np.average(self.particles[:, 0:2], weights=self.weights, axis=0)



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



if __name__ == '__main__':

    DO_PLOT_PARTICLES = False
    from numpy.random import seed
    import matplotlib.pyplot as plt

    #plt.figure()

    seed(5)
    for count in range(10):
        print()
        print(count)
        #numpy.random.set_state(fail_state)
        #if count == 12:
        #    #fail_state = numpy.random.get_state()
        #    DO_PLOT_PARTICLES = True

        N = 4000
        sensor_std_err = .1
        landmarks = np.array([[-1, 2], [2,4], [10,6], [18,25]])
        NL = len(landmarks)

        #landmarks = [[-1, 2], [2,4]]

        pf = RobotLocalizationParticleFilter(N, 20, 20, landmarks, sensor_std_err)
        #pf.create_gaussian_particles([3, 2, 0], [5, 5, 2])
        pf.create_uniform_particles((0,20), (0,20), (0, 6.28))

        if DO_PLOT_PARTICLES:
            plt.scatter(pf.particles[:, 0], pf.particles[:, 1], alpha=.2, color='g')

        xs = []
        for x in range(18):
            zs = []
            pos=(x+1, x+1)

            for landmark in landmarks:
                d = np.sqrt((landmark[0]-pos[0])**2 +  (landmark[1]-pos[1])**2)
                zs.append(d + randn()*sensor_std_err)


            zs = np.linalg.norm(landmarks - pos, axis=1) + randn(NL)*sensor_std_err


            # move diagonally forward to (x+1, x+1)
            pf.predict((0.00, 1.414), (.2, .05))

            pf.update(z=zs)
            if x == 0:
                print(max(pf.weights))
            #while abs(pf.neff() -N) < .1:
            #    print('neffing')
            #    pf.create_uniform_particles((0,20), (0,20), (0, 6.28))
            #    pf.update(z=zs)
            #print(pf.neff())
            #indexes = residual_resample2(pf.weights)
            indexes = systemic_resample(pf.weights)

            pf.resample_from_index(indexes)
            #pf.resample()

            mu, var = pf.estimate()
            xs.append(mu)
            if DO_PLOT_PARTICLES:
                plt.scatter(pf.particles[:, 0], pf.particles[:, 1], alpha=.2)
                plt.scatter(pos[0], pos[1], marker='*', color='r')
                plt.scatter(mu[0], mu[1], marker='s', color='r')
                plt.pause(.01)

        xs = np.array(xs)
        plt.plot(xs[:, 0], xs[:, 1])
        plt.show()
