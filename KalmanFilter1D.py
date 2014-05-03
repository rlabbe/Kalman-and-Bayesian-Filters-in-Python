# -*- coding: utf-8 -*-
"""
Created on Thu May  1 19:48:54 2014

@author: rlabbe
"""
from __future__ import division
import gauss as g;
import math
import matplotlib.pyplot as plt
import noise
import numpy.random as random


class KalmanFilter1D(object):

    def __init__ (self, est_0=g.gaussian(0,1000)):
        self.estimate = est_0

    def update_estimate(self, Z):
        self.estimate = self.estimate * Z

    def project_ahead(self, U):
        self.estimate = self.estimate + U



def fixed_error_kf(measurement_error, motion_error, noise_factor = 1.0):
    mu = 0
    sig = 1000
    measurements = [x+5 for x in range(100)]
    f = KalmanFilter1D (g.gaussian(mu,sig))

    ys = []
    errs = []
    xs = []

    for i in range(len(measurements)):
        r = noise.white_noise (noise_factor)
        m = measurements[i] + r
        f.update_estimate (g.gaussian(m, measurement_error))

        xs.append(m)
        ys.append(f.estimate.mu)
        errs.append (f.estimate.sigma)

        f.project_ahead (g.gaussian(20, motion_error))

    plt.clf()

    p1, = plt.plot (measurements, 'r')
    p2, = plt.plot (xs,'g')
    p3, = plt.plot (ys, 'b')
    plt.legend ([p1,p2,p3],['actual', 'measurement', 'filter'], 2)
    #plt.errorbar (x=range(len(ys)), color='b', y=ys, yerr=errs)
    plt.show()

def varying_error_kf(noise_factor=1.0):
    motion_sig = 2.
    mu = 0
    sig = 1000


    f = KF1D (mu,sig)
    ys = []
    us = []
    errs = []
    xs = []

    for i in range(len(measurements)):
        r = random.randn() * noise_factor
        m = measurements[i] + r
        print (r)
        f.update (m, abs(r*10))
        xs.append(m)
        #print ("measure:" + str(f.estimate))
        ys.append(f.estimate.mu)
        errs.append (f.estimate.sigma)

        f.predict (1.0, motion_sig)
        #print ("predict:" + str(f.estimate))

    plt.clf()
    plt.plot (measurements, 'r')
    plt.plot (xs,'g')
    plt.errorbar (x=range(len(ys)), color='b', y=ys, yerr=errs)
    plt.show()


def _test_foo ():
    sensor_error = 1
    movement = .1
    movement_error = .1
    pos = g.gaussian(1,500)

    zs = []
    ps = []
    filter_ = KalmanFilter1D(pos)
    m_1 = filter_.estimate.mu

    for i in range(300):
        filter_.update_estimate(g.gaussian(movement, movement_error))

        Z = math.sin(i/12.) + math.sqrt(abs(noise.white_noise(.02)))
        movement = filter_.estimate.mu - m_1
        m_1 = filter_.estimate.mu

        print movement, filter_.estimate.sigma
        zs.append(Z)

        filter_.project_ahead (g.gaussian(Z, sensor_error))
        ps.append(filter_.estimate[0])


    p1, = plt.plot(zs,c='r', linestyle='dashed')
    p2, = plt.plot(ps, c='b')
    plt.legend([p1,p2], ['measurement', 'filter'], 2)
    plt.show()

if __name__ == "__main__":

    if 1:
        # use same seed to get repeatable results
        random.seed(10)

    fixed_error_kf(measurement_error=100.,
                   motion_error=.1,
                   noise_factor=50)
    #_test_foo()