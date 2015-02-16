# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 09:34:36 2015

@author: rlabbe
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import array, asarray
from numpy.random import randn
import math
from filterpy.kalman import UnscentedKalmanFilter as UKF



class RadarSim(object):
    """ Simulates the radar signal returns from an object flying
    at a constant altityude and velocity in 1D.
    """

    def __init__(self, dt, pos, vel, alt):
        self.pos = pos
        self.vel = vel
        self.alt = alt
        self.dt = dt

    def get_range(self):
        """ Returns slant range to the object. Call once for each
        new measurement at dt time from last call.
        """

        # add some process noise to the system
        self.vel = self.vel  + .1*randn()
        self.alt = self.alt + .1*randn()
        self.pos = self.pos + self.vel*self.dt

        # add measurment noise
        err = self.pos * 0.05*randn()
        slant_dist = math.sqrt(self.pos**2 + self.alt**2)

        return slant_dist + err



dt = 0.05


def hx(x):
    return (x[0]**2 + x[2]**2)**.5
    pass



def fx(x, dt):
    result = x.copy()
    result[0] += x[1]*dt
    return result




f = UKF(3, 1, dt= dt, hx=hx, fx=fx, kappa=1)
radar = RadarSim(dt, pos=-1000., vel=100., alt=1000.)

f.x = array([0, 90, 1005])
f.R = 0.1
f.Q *= 0.002




xs = []
track = []

for i in range(int(20/dt)):
    z = radar.get_range()
    track.append((radar.pos, radar.vel, radar.alt))

    f.predict()
    f.update(array([z]))

    xs.append(f.x)


xs = asarray(xs)
track = asarray(track)
time = np.arange(0,len(xs)*dt, dt)

plt.figure()
plt.subplot(311)
plt.plot(time, track[:,0])
plt.plot(time, xs[:,0])
plt.legend(loc=4)
plt.xlabel('time (sec)')
plt.ylabel('position (m)')


plt.subplot(312)
plt.plot(time, track[:,1])
plt.plot(time, xs[:,1])
plt.legend(loc=4)
plt.xlabel('time (sec)')
plt.ylabel('velocity (m/s)')

plt.subplot(313)
plt.plot(time, track[:,2])
plt.plot(time, xs[:,2])
plt.ylabel('altitude (m)')
plt.legend(loc=4)
plt.xlabel('time (sec)')
plt.ylim((900,1600))
plt.show()
