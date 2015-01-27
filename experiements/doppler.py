# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 15:04:08 2015

@author: rlabbe
"""



from numpy.random import randn
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from math import sqrt
import math
import random

pos_var = 1.
dop_var = 2.
dt = 1/20


def rand_student_t(df, mu=0, std=1):
    """return random number distributed by student's t distribution with
    `df` degrees of freedom with the specified mean and standard deviation.
    """
    x = random.gauss(0, std)
    y = 2.0*random.gammavariate(0.5*df, 2.0)
    return x / (math.sqrt(y/df)) + mu
    


np.random.seed(124)
class ConstantVelocityObject(object):
    def __init__(self, x0, vel, noise_scale):
        self.x = np.array([x0, vel])
        self.noise_scale = noise_scale
        self.vel = vel


    def update(self, dt):
        pnoise = abs(randn()*self.noise_scale)
        if self.x[1] > self.vel:
            pnoise = -pnoise
            
        self.x[1] += pnoise
        self.x[0] += self.x[1]*dt

        return self.x.copy()
        
        
    def sense_pos(self):
        return self.x[0] + [randn()*self.noise_scale[0]]

        
    def vel(self):
        return self.x[1] + [randn()*self.noise_scale[1]]


def sense(x, noise_scale=(1., 0.01)):
    return x + [randn()*noise_scale[0], randn()*noise_scale[1]]

def sense_t(x, noise_scale=(1., 0.01)):
    return x + [rand_student_t(2)*noise_scale[0], 
                rand_student_t(2)*noise_scale[1]]





R2 = np.zeros((2, 2))
R1 = np.zeros((1, 1))

R2[0, 0] = pos_var
R2[1, 1] = dop_var

R1[0,0] = dop_var

H2 = np.array([[1., 0.], [0., 1.]])
H1 = np.array([[0., 1.]])



kf = KalmanFilter(2, 1)
kf.F = array([[1, dt], 
              [0, 1]])
kf.H = array([[1, 0], 
              [0, 1]])
kf.Q = Q_discrete_white_noise(2, dt, .02)
kf.x = array([0., 0.])
kf.R = R1
kf.H = H1
kf.P *= 10


random.seed(1234)
track = []
zs = []
ests = []
sensor_noise = (sqrt(pos_var), sqrt(dop_var))
obj = ConstantVelocityObject(0., 1., noise_scale=.18)

for i in range(30000):
    x = obj.update(dt/10)
    z = sense_t(x, sensor_noise)
    if i % 10 != 0:
        continue

    if i % 60 == 87687658760:
        kf.predict()
        kf.update(z, H=H2, R=R2)        
    else:
        kf.predict()
        kf.update(z[1], H=H1, R=R1) 
        
    ests.append(kf.x)
        
    track.append(x)
    zs.append(z)


track = np.asarray(track)
zs = np.asarray(zs)
ests = np.asarray(ests)

plt.figure()
plt.subplot(211)
plt.plot(track[:,0], label='track')
plt.plot(zs[:,0], label='measurements')
plt.plot(ests[:,0], label='filter')
plt.legend(loc='best')

plt.subplot(212)
plt.plot(track[:,1], label='track')
plt.plot(zs[:,1], label='measurements')
plt.plot(ests[:,1], label='filter')
plt.show()

