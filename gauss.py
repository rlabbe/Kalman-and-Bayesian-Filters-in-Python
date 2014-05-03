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







#varying_error_kf( noise_factor=100)
