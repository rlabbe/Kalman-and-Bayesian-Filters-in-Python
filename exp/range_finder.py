# -*- coding: utf-8 -*-
"""
Created on Sat May 24 19:14:06 2014

@author: rlabbe
"""
from __future__ import division, print_function
from KalmanFilter import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
import math


class DMESensor(object):
    def __init__(self, pos_a, pos_b, noise_factor=1.0):
        self.A = pos_a
        self.B = pos_b
        self.noise_factor = noise_factor

    def range_of (self, pos):
        """ returns tuple containing noisy range data to A and B
        given a position 'pos'
        """

        ra = math.sqrt((self.A[0] - pos[0])**2 + (self.A[1] - pos[1])**2)
        rb = math.sqrt((self.B[0] - pos[0])**2 + (self.B[1] - pos[1])**2)

        return (ra + random.randn()*self.noise_factor,
                rb + random.randn()*self.noise_factor)


def dist(a,b):
    return math.sqrt ((a[0]-b[0])**2 + (a[1]-b[1])**2)

def H_of (pos, pos_A, pos_B):
    from math import sin, cos, atan2

    theta_a = atan2(pos_a[1]-pos[1], pos_a[0] - pos[0])
    theta_b = atan2(pos_b[1]-pos[1], pos_b[0] - pos[0])

    return np.mat([[-cos(theta_a), 0, -sin(theta_a), 0],
                   [-cos(theta_b), 0, -sin(theta_b), 0]])

    # equivalently we can do this...
    #dist_a = dist(pos, pos_A)
    #dist_b = dist(pos, pos_B)

    #return np.mat([[(pos[0]-pos_A[0])/dist_a, 0, (pos[1]-pos_A[1])/dist_a,0],
    #               [(pos[0]-pos_B[0])/dist_b, 0, (pos[1]-pos_B[1])/dist_b,0]])




pos_a = (100,-20)
pos_b = (-100, -20)

f1 = KalmanFilter(dim=4)
dt = 1.0   # time step
'''
f1.F = np.mat ([[1, dt, 0,  0],
                [0,  1, 0,  0],
                [0,  0, 1, dt],
                [0,  0, 0,  1]])

'''
f1.F = np.mat ([[0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0]])

f1.B = 0.

f1.R = np.eye(2) * 1.
f1.Q = np.eye(4) * .1

f1.x = np.mat([1,0,1,0]).T
f1.P = np.eye(4) * 5.

# initialize storage and other variables for the run
count = 30
xs, ys = [],[]
pxs, pys = [],[]

d = DMESensor (pos_a, pos_b, noise_factor=1.)

pos = [0,0]
for i in range(count):
    pos = (i,i)
    ra,rb = d.range_of(pos)
    rx,ry = d.range_of((i+f1.x[0,0], i+f1.x[2,0]))

    print ('range =', ra,rb)

    z = np.mat([[ra-rx],[rb-ry]])
    print('z =', z)

    f1.H = H_of (pos, pos_a, pos_b)
    print('H =', f1.H)

    ##f1.update (z)

    print (f1.x)
    xs.append (f1.x[0,0]+i)
    ys.append (f1.x[2,0]+i)
    pxs.append (pos[0])
    pys.append(pos[1])
    f1.predict ()
    print (f1.H * f1.x)
    print (z)
    print (f1.x)
    f1.update(z)
    print(f1.x)

p1, = plt.plot (xs, ys, 'r--')
p2, = plt.plot (pxs, pys)
plt.legend([p1,p2], ['filter', 'ideal'], 2)
plt.show()
