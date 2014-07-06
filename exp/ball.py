# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 16:07:29 2014

@author: rlabbe
"""

import numpy as np
from KalmanFilter import KalmanFilter
from math import radians, sin, cos
import numpy.random as random
import matplotlib.markers as markers

class BallPath(object):
    def __init__(self, x0, y0, omega_deg, velocity, g=9.8, noise=[1.0,1.0]):
        omega = radians(omega_deg)
        self.vx0 = velocity * cos(omega)
        self.vy0 = velocity * sin(omega)
        
        self.x0 = x0
        self.y0 = y0
        
        self.g = g
        self.noise = noise
        
    def pos_at_t(self, t):
        """ returns (x,y) tuple of ball position at time t"""
        x = self.vx0*t + self.x0
        y = -0.5*self.g*t**2 + self.vy0*t + self.y0
        
        return (x +random.randn()*self.noise[0], y +random.randn()*self.noise[1])
        
        
        

y = 15
x = 0
omega = 0.
noise = [1,1]
v0 = 100.
ball = BallPath (x0=x, y0=y, omega_deg=omega, velocity=v0, noise=noise)
t = 0
dt = 1
g = 9.8


f1 = KalmanFilter(dim=6)
dt = 1/30.   # time step

ay = -.5*dt**2

f1.F = np.mat ([[1, dt,  0,  0,  0,  0],   # x=x0+dx*dt
                [0,  1, dt,  0,  0,  0],   # dx = dx
                [0,  0,  0,  0,  0,  0],   # ddx = 0
                [0,  0,  0,  1, dt, ay],   # y = y0 +dy*dt+1/2*g*dt^2
                [0,  0,  0,  0,  1, dt],   # dy = dy0 + ddy*dt 
                [0,  0,  0,  0,  0, 1]])  # ddy = -g
f1.B = 0.

f1.H = np.mat([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]])

f1.R = np.eye(2) * 5
f1.Q = np.eye(6) * 0.

omega = radians(omega)
vx = cos(omega) * v0
vy = sin(omega) * v0

f1.x = np.mat([x,vx,0,y,vy,-9.8]).T

f1.P = np.eye(6) * 500.




z = np.mat([[0,0]]).T
count = 0
markers.MarkerStyle(fillstyle='none')

np.set_printoptions(precision=4)
while f1.x[3,0] > 0:
    count += 1
    #f1.update (z)
    f1.predict()
    print f1.x[0,0], f1.x[3,0]
    #markers.set_fillstyle('none')
    plt.scatter(f1.x[0,0],f1.x[3,0], color='green')
  
    
    



