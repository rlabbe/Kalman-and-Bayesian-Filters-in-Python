# -*- coding: utf-8 -*-
"""
Created on Thu May  1 19:48:54 2014

@author: rlabbe
"""

import math
import matplotlib.pyplot as plt
import noise
import numba
import numpy.random as random

class KalmanFilter1D(object):

    def __init__ (self, x0, var):
        self.mean = x0
        self.variance = var
    
    def estimate(self, z, z_variance):
        self.mean = (self.variance*z + z_variance*self.mean) / (self.variance + z_variance)
        self.variance = 1. / (1./self.variance + 1./z_variance)

    def project(self, u, u_variance):
        self.mean += u
        self.variance += u_variance


def _fixed_error_kf(measurement_error, motion_error, noise_factor = 1.0):
    mean = 0
    sig = 1000
    measurements = [x for x in range(100)]
    f = KalmanFilter1D (mean,sig)

    ys = []
    errs = []
    xs = []

    for i in range(len(measurements)):
        r = noise.white_noise (noise_factor)
        z = measurements[i] + r
        f.estimate (z, measurement_error)

        xs.append(z)
        ys.append(f.mean)
        errs.append (f.variance)

        f.project (1, motion_error)

    plt.clf()

    p1, = plt.plot (measurements, 'r')
    p2, = plt.plot (xs,'g')
    p3, = plt.plot (ys, 'b')
    plt.legend ([p1,p2,p3],['actual', 'measurement', 'filter'], 2)
    #plt.errorbar (x=range(len(ys)), color='b', y=ys, yerr=errs)
    plt.show()

def _varying_error_kf(noise_factor=1.0):
    motion_sig = 2.
    mean = 0
    sig = 1000

    measurements = [x for x in range(100)]

    f = KalmanFilter1D (mean,sig)
    ys = []
    errs = []
    xs = []

    for i in range(len(measurements)):
        r = random.randn() * noise_factor
        m = measurements[i] + r

        f.estimate (m, abs(r*10))
        xs.append(m)
        ys.append(f.mean)
        errs.append (f.variance)

        f.project (1.0, motion_sig)

    plt.clf()
    plt.plot (measurements, 'r')
    plt.plot (xs,'g')
    plt.errorbar (x=range(len(ys)), color='b', y=ys, yerr=errs)
    plt.show()



def _test_sin ():
    sensor_error = 1
    movement = .1
    movement_error = .1
    pos = (1,500)

    zs = []
    ps = []
    filter_ = KalmanFilter1D(pos[0],pos[1])
    m_1 = filter_.mean

    for i in range(300):
        filter_.project(movement, movement_error)

        Z = math.sin(i/12.) + math.sqrt(abs(noise.white_noise(.02)))
        movement = filter_.mean - m_1
        m_1 = filter_.mean

        zs.append(Z)

        filter_.estimate (Z, sensor_error)
        ps.append(filter_.mean)


    p1, = plt.plot(zs,c='r', linestyle='dashed')
    p2, = plt.plot(ps, c='b')
    plt.legend([p1,p2], ['measurement', 'filter'], 2)
    plt.show()

if __name__ == "__main__":

    if 0:
        # use same seed to get repeatable results
        random.seed(10)

    if 1:
        plt.figure()
        _varying_error_kf ()
    if 1:
        plt.figure()
        _fixed_error_kf(measurement_error=1.,
                        motion_error=.1,
                        noise_factor=50)

    if 1:
        plt.figure()
        _test_sin()
