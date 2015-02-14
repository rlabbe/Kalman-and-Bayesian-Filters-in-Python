# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 17:47:56 2015

@author: rlabbe
"""
import numpy as np
from filterpy.kalman import  UnscentedKalmanFilter as UKF
from math import atan2, radians,degrees
from filterpy.common import stats
import matplotlib.pyplot as plt

p = (-10, -10)  

def hx(x):
    dx = x[0] - hx.p[0]
    dy = x[1] - hx.p[1]    
    return np.array([atan2(dy,dx), (dx**2 + dy**2)**.5])
    
def fx(x,dt):
    return x
    


kf = UKF(2, 2, dt=0.1, hx=hx, fx=fx, kappa=2.)

kf.x = np.array([100, 100.])
kf.P *= 40
hx.p = kf.x - np.array([50,50])

d = ((kf.x[0] - hx.p[0])**2 + (kf.x[1] - hx.p[1])**2)**.5

stats.plot_covariance_ellipse(
       kf.x, cov=kf.P, axis_equal=True, 
       facecolor='y', edgecolor=None, alpha=0.6)
plt.scatter([100], [100], c='y', label='Initial')

kf.R[0,0] = radians (1)**2
kf.R[1,1] = 2.**2


kf.predict()
kf.update(np.array([radians(45), d]))

print(kf.x)
print(kf.P)

stats.plot_covariance_ellipse(
       kf.x, cov=kf.P, axis_equal=True, 
       facecolor='g', edgecolor=None, alpha=0.6)
plt.scatter([100], [100], c='g', label='45 degrees')
       
       
p = (13, -11)
hx.p = kf.x - np.array([-50,50])
d = ((kf.x[0] - hx.p[0])**2 + (kf.x[1] - hx.p[1])**2)**.5

kf.predict()
kf.update(np.array([radians(135), d]))

stats.plot_covariance_ellipse(
       kf.x, cov=kf.P, axis_equal=True, 
       facecolor='b', edgecolor=None, alpha=0.6)
plt.scatter([100], [100], c='b', label='135 degrees')

plt.legend(scatterpoints=1, markerscale=3)