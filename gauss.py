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
    def __init__(self, mean, variance):
        try:
            self.mean     = float(mean)
            self.variance = float(variance)

        except:
            self.mean     = mean
            self.variance = variance

    def __add__ (a, b):
       return gaussian (a.mean + b.mean, a.variance + b.variance)

    def __mul__ (a, b):
        m = (a.variance*b.mean + b.variance*a.mean) / (a.variance + b.variance)
        v = 1. / (1./a.variance + 1./b.variance)
        return gaussian (m, v)

    def __call__(self, x):
        """ Impl
        """
        return math.exp (-0.5 * (x-self.mean)**2 / self.variance) / \
               math.sqrt(2.*math.pi*self.variance)


    def __str__(self):
        return "(%f, %f)" %(self.mean, self.sigma)

    def stddev(self):
        return math.sqrt (self.variance)

    def as_tuple(self):
        return (self.mean, self.variance)

    def __tuple__(self):
        return (self.mean, self.variance)

    def __getitem__ (self,index):
        """ maybe silly, allows you to access obect as a tuple:
        a = gaussian(3,4)
        print (tuple(a))
        """
        if index == 0:   return self.mean
        elif index == 1: return self.variance
        else:            raise StopIteration

class KF1D(object):
    def __init__ (self, pos, sigma):
        self.estimate = gaussian(pos,sigma)

    def update(self, Z,var):
        self.estimate = self.estimate * gaussian (Z,var)

    def predict(self, U, var):
        self.estimate = self.estimate + gaussian (U,var)




def mul2 (a, b):
    m = (a['variance']*b['mean'] + b['variance']*a['mean']) / (a['variance'] + b['variance'])
    v = 1. / (1./a['variance'] + 1./b['variance'])
    return gaussian (m, v)


#varying_error_kf( noise_factor=100)
