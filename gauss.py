# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:38:49 2014

@author: rlabbe
"""
from __future__ import division, print_function
import math
import matplotlib.pyplot as plt
import numpy.random as random


class gaussian(object):
    def __init__ (self,m,s):
        self.mu    = float(m)
        self.sigma = float(s)

    def __add__ (a,b):
       return gaussian (a.mu + b.mu, a.sigma + b.sigma)

    def __mul__ (a,b):
        m = (a.sigma*b.mu + b.sigma*a.mu) / (a.sigma + b.sigma)
        s = 1. / (1./a.sigma + 1./b.sigma)
        return gaussian (m,s)

    def __call__(self,x):
        return math.exp (-0.5 * (x-self.mu)**2 / self.sigma) / \
        math.sqrt(2.*math.pi*self.sigma)

    def __str__(self):
        return "(%f, %f)" %(self.mu, self.sigma)

    def __getitem__ (self,index):
        """ maybe silly, allows you to access obect as a tuple:
        a = gaussian(3,4)
        print (tuple(a))
        """
        if index == 0:   return self.mu
        elif index == 1: return self.sigma
        else:            raise StopIteration

class KF1D(object):
    def __init__ (self, pos, sigma):
        self.estimate = gaussian(pos,sigma)

    def update(self, Z,var):
        self.estimate = self.estimate * gaussian (Z,var)

    def predict(self, U, var):
        self.estimate = self.estimate + gaussian (U,var)


measurements = [x+5 for x in range(100)]



def fixed_error_kf(measurement_error, noise_factor = 1.0):
    motion_sig = 2.
    mu = 0
    sig = 1000

    f = KF1D (mu,sig)

    ys = []
    errs = []
    xs = []

    for i in range(len(measurements)):
        r = random.randn() * noise_factor
        m = measurements[i] + r
        f.update (m, measurement_error)

        xs.append(m)
        ys.append(f.estimate.mu)
        errs.append (f.estimate.sigma)

        f.predict (1.0, motion_sig)

    plt.clf()

    plt.plot (measurements, 'r')
    plt.plot (xs,'g')
    plt.errorbar (x=range(len(ys)), color='b', y=ys, yerr=errs)
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

varying_error_kf( noise_factor=100)
