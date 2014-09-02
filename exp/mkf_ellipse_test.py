# -*- coding: utf-8 -*-
"""
Created on Sun May 11 20:47:52 2014

@author: rlabbe
"""

from DogSensor import DogSensor
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import stats

def dog_tracking_filter(R,Q=0,cov=1.):
    f = KalmanFilter (dim_x=2, dim_z=1)
    f.x = np.matrix([[0], [0]])    # initial state (location and velocity)
    f.F = np.matrix([[1,1],[0,1]]) # state transition matrix
    f.H = np.matrix([[1,0]])       # Measurement function
    f.R = R                        # measurement uncertainty
    f.P *= cov                     # covariance matrix
    f.Q = Q
    return f


def plot_track(noise, count, R, Q=0, plot_P=True, title='Kalman Filter'):
    dog = DogSensor(velocity=1, noise=noise)
    f = dog_tracking_filter(R=R, Q=Q, cov=10.)

    ps = []
    zs = []
    cov = []
    for t in range (count):
        z = dog.sense()
        f.update (z)
        #print (t,z)
        ps.append (f.x[0,0])
        cov.append(f.P)
        zs.append(z)
        f.predict()

    p0, = plt.plot([0,count],[0,count],'g')
    p1, = plt.plot(range(1,count+1),zs,c='r', linestyle='dashed')
    p2, = plt.plot(range(1,count+1),ps, c='b')
    plt.axis('equal')
    plt.legend([p0,p1,p2], ['actual','measurement', 'filter'], 2)
    plt.title(title)

    for i,p in enumerate(cov):
        print(i,p)
        e = stats.sigma_ellipse (p, i+1, ps[i])
        stats.plot_sigma_ellipse(e, axis_equal=False)
    plt.xlim((-1,count))
    plt.show()


if __name__ == "__main__":
    plot_track (noise=30, R=5, Q=2, count=20)