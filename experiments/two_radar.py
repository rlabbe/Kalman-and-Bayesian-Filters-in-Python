# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 14:29:23 2015

@author: rlabbe
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 09:34:36 2015

@author: rlabbe
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import array, asarray
from numpy.linalg import norm
from numpy.random import randn
import math
from math import atan2, radians, degrees


from filterpy.kalman import UnscentedKalmanFilter as UKF


class RadarStation(object):
    
    def __init__(self, pos, range_std, bearing_std):
        self.pos = asarray(pos)
        
        self.range_std = range_std
        self.bearing_std = bearing_std

    
    def reading_of(self, ac_pos):
        """ Returns range and bearing to aircraft as tuple. bearing is in 
        radians.
        """
        
        diff = np.subtract(self.pos, ac_pos)
        rng = norm(diff)
        brg = atan2(diff[1], diff[0])      
        return rng, brg


    def noisy_reading(self, ac_pos):
        rng, brg = self.reading_of(ac_pos)      
        rng += randn() * self.range_std
        brg += randn() * self.bearing_std 
        return rng, brg
        
        
    
    
class ACSim(object):
    
    def __init__(self, pos, vel, vel_std):
        self.pos = asarray(pos, dtype=float)
        self.vel = asarray(vel, dtype=float)
        self.vel_std = vel_std
        
        
    def update(self):
        vel = self.vel + (randn() * self.vel_std)       
        self.pos += vel
        print(pos)
        
        return self.pos



dt = 1.


def hx(x):
    r1, b1 = hx.R1.reading_of((x[0], x[2]))
    r2, b2 = hx.R2.reading_of((x[0], x[2]))
    
    return array([r1, b1, r2, b2])
    pass



def fx(x, dt):
    x_est = x.copy()
    x_est[0] += x[1]*dt
    x_est[2] += x[3]*dt
    return x_est




f = UKF(dim_x=4, dim_z=4, dt=dt, hx=hx, fx=fx, kappa=1)
aircraft = ACSim ((100,100), (0.1,0.1), 0.0)

R1 = RadarStation ((0,0), range_std=1.0, bearing_std=radians(0.5))
R2 = RadarStation ((200,0), range_std=1.0, bearing_std=radians(0.5))

hx.R1 = R1
hx.R2 = R2

f.x = array([100, 1, 100, 1])
f.R = np.diag([1.0, 0.5, 1.0, 0.5])
f.Q *= 0.0020




xs = []
track = []

for i in range(int(20/dt)):
    
    pos = aircraft.update()
    
    r1, b1 = R1.noisy_reading(pos)
    r2, b2, = R2.noisy_reading(pos)
    
    z = np.array([r1, b1, r2, b2])
    track.append(pos.copy())

    f.predict()
    f.update(z)
    xs.append(f.x)


xs = asarray(xs)
track = asarray(track)
time = np.arange(0,len(xs)*dt, dt)

plt.figure()
plt.subplot(411)
plt.plot(time, track[:,0])
plt.plot(time, xs[:,0])
plt.legend(loc=4)
plt.xlabel('time (sec)')
plt.ylabel('x position (m)')



plt.subplot(412)
plt.plot(time, track[:,1])
plt.plot(time, xs[:,3])
plt.legend(loc=4)
plt.xlabel('time (sec)')
plt.ylabel('y position (m)')


plt.subplot(413)
plt.plot(time, xs[:,1])
plt.legend(loc=4)
plt.xlabel('time (sec)')
plt.ylabel('x velocity (m/s)')

plt.subplot(414)
plt.plot(time, xs[:,3])
plt.ylabel('y velocity (m/s)')
plt.legend(loc=4)
plt.xlabel('time (sec)')

plt.show()
