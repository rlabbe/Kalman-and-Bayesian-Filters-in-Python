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
from math import sin, cos, atan2, radians, degrees


from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise

class RadarStation(object):

    def __init__(self, pos, range_std, bearing_std):
        self.pos = asarray(pos)

        self.range_std = range_std
        self.bearing_std = bearing_std


    def reading_of(self, ac_pos):
        """ Returns range and bearing to aircraft as tuple. bearing is in
        radians.
        """

        diff = np.subtract(ac_pos, self.pos)
        rng = norm(diff)
        brg = atan2(diff[1], diff[0])
        return rng, brg


    def noisy_reading(self, ac_pos):
        rng, brg = self.reading_of(ac_pos)
        rng += randn() * self.range_std
        brg += randn() * self.bearing_std
        return rng, brg


    def z_to_x(self, slant_range, angle):
        """ given a reading, convert to world coordinates"""
        
        x = cos(angle)*slant_range
        z = sin(angle)*slant_range
        
        return self.pos + (x,z)
        


class ACSim(object):

    def __init__(self, pos, vel, vel_std):
        self.pos = asarray(pos, dtype=float)
        self.vel = asarray(vel, dtype=float)
        self.vel_std = vel_std

    def update(self):
        vel = self.vel + (randn() * self.vel_std)
        self.pos += vel

        return self.pos


def two_radar_constvel():
    dt = 5
    
    
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
    
    
    
    
    f = UKF(dim_x=4, dim_z=4, dt=dt, hx=hx, fx=fx, kappa=0)
    aircraft = ACSim ((100,100), (0.1*dt,0.02*dt), 0.002)
    
    
    range_std = 0.2
    bearing_std = radians(0.5)
    
    R1 = RadarStation ((0,0), range_std, bearing_std)
    R2 = RadarStation ((200,0), range_std, bearing_std)
    
    hx.R1 = R1
    hx.R2 = R2
    
    f.x = array([100, 0.1, 100, 0.02])
    f.R = np.diag([range_std**2, bearing_std**2, range_std**2, bearing_std**2])
    q = Q_discrete_white_noise(2, var=0.002, dt=dt)
    #q = np.array([[0,0],[0,0.0002]])
    f.Q[0:2, 0:2] = q
    f.Q[2:4, 2:4] = q
    f.P = np.diag([.1, 0.01, .1, 0.01])
    
    
    track = []
    zs = []
    
    
    for i in range(int(300/dt)):
    
        pos = aircraft.update()
    
        r1, b1 = R1.noisy_reading(pos)
        r2, b2 = R2.noisy_reading(pos)
    
        z = np.array([r1, b1, r2, b2])
        zs.append(z)
        track.append(pos.copy())
    
    zs = asarray(zs)
    
    
    xs, Ps, Pxz = f.batch_filter(zs)
    ms, _, _ = f.rts_smoother2(xs, Ps, Pxz)
    
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
    plt.plot(time, xs[:,2])
    plt.legend(loc=4)
    plt.xlabel('time (sec)')
    plt.ylabel('y position (m)')
    
    
    plt.subplot(413)
    plt.plot(time, xs[:,1])
    plt.plot(time, ms[:,1])
    plt.legend(loc=4)
    plt.ylim([0, 0.2])
    plt.xlabel('time (sec)')
    plt.ylabel('x velocity (m/s)')
    
    plt.subplot(414)
    plt.plot(time, xs[:,3])
    plt.plot(time, ms[:,3])
    plt.ylabel('y velocity (m/s)')
    plt.legend(loc=4)
    plt.xlabel('time (sec)')
    
    plt.show()


def two_radar_constalt():
    dt = .05
    
    
    def hx(x):
        r1, b1 = hx.R1.reading_of((x[0], x[2]))
        r2, b2 = hx.R2.reading_of((x[0], x[2]))
    
        return array([r1, b1, r2, b2])
        pass
    
    
    def fx(x, dt):
        x_est = x.copy()
        x_est[0] += x[1]*dt
        return x_est
    
   
    
    vx = 100/1000 # meters/sec
    vz = 0.
    
    f = UKF(dim_x=3, dim_z=4, dt=dt, hx=hx, fx=fx, kappa=0)
    aircraft = ACSim ((0, 1), (vx*dt, vz*dt), 0.00)
    
    
    range_std = 1/1000.
    bearing_std =1/1000000.
    
    R1 = RadarStation ((  0, 0), range_std, bearing_std)
    R2 = RadarStation ((60, 0), range_std, bearing_std)
    
    hx.R1 = R1
    hx.R2 = R2
    
    f.x = array([aircraft.pos[0], vx, aircraft.pos[1]])
    f.R = np.diag([range_std**2, bearing_std**2, range_std**2, bearing_std**2])
    q = Q_discrete_white_noise(2, var=0.0002, dt=dt)
    #q = np.array([[0,0],[0,0.0002]])
    f.Q[0:2, 0:2] = q
    f.Q[2,2] = 0.0002
    f.P = np.diag([.1, 0.01, .1])*0.1
    
    
    track = []
    zs = []
    
    
    for i in range(int(500/dt)):   
        pos = aircraft.update()
    
        r1, b1 = R1.noisy_reading(pos)
        r2, b2 = R2.noisy_reading(pos)
    
        z = np.array([r1, b1, r2, b2])
        zs.append(z)
        track.append(pos.copy())
    
    zs = asarray(zs)
    
    
    xs, Ps = f.batch_filter(zs)
    ms, _, _ = f.rts_smoother(xs, Ps)
    
    track = asarray(track)
    time = np.arange(0,len(xs)*dt, dt)
    
    plt.figure()
    plt.subplot(311)
    plt.plot(time, track[:,0])
    plt.plot(time, xs[:,0])
    plt.legend(loc=4)
    plt.xlabel('time (sec)')
    plt.ylabel('x position (m)')  

    plt.subplot(312)
    plt.plot(time, xs[:,1]*1000, label="UKF")
    plt.plot(time, ms[:,1]*1000, label='RTS')
    plt.legend(loc=4)
    plt.xlabel('time (sec)')
    plt.ylabel('velocity (m/s)')
    
    plt.subplot(313)
    plt.plot(time, xs[:,2]*1000, label="UKF")
    plt.plot(time, ms[:,2]*1000, label='RTS')
    plt.legend(loc=4)
    plt.xlabel('time (sec)')
    plt.ylabel('altitude (m)')
    plt.ylim([900,1100])
    
    for z in zs[:10]:
        p = R1.z_to_x(z[0], z[1])
        #plt.scatter(p[0], p[1], marker='+', c='k')
        
        p = R2.z_to_x(z[2], z[3])
        #plt.scatter(p[0], p[1], marker='+', c='b')
  
    plt.show()


two_radar_constalt()