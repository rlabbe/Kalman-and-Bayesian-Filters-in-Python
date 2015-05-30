# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:18:54 2015

@author: rlabbe
"""


from math import cos, sin, sqrt, atan2
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, dot
from numpy.linalg import pinv
from numpy.random import randn
from filterpy.common import plot_covariance_ellipse
from filterpy.kalman import ExtendedKalmanFilter as EKF


def print_x(x):
    print(x[0, 0], x[1, 0], np.degrees(x[2, 0]))


def normalize_angle(x, index):
    if x[index] > np.pi:
        x[index] -= 2*np.pi
    if x[index] < -np.pi:
        x[index] = 2*np.pi

def residual(a,b):
    y = a - b
    normalize_angle(y, 1)
    return y


def control_update(x, u, dt):
    """ x is [x, y, hdg], u is [vel, omega] """

    v = u[0]
    w = u[1]
    if w == 0:
        # approximate straight line with huge radius
        w = 1.e-30
    r = v/w # radius


    return x + np.array([[-r*sin(x[2]) + r*sin(x[2] + w*dt)],
                         [ r*cos(x[2]) - r*cos(x[2] + w*dt)],
                         [w*dt]])

a1 = 0.001
a2 = 0.001
a3 = 0.001
a4 = 0.001

sigma_r = 0.1
sigma_h =     a_error = np.radians(1)
sigma_s = 0.00001

def normalize_angle(x, index):
    if x[index] > np.pi:
        x[index] -= 2*np.pi
    if x[index] < -np.pi:
        x[index] = 2*np.pi



class RobotEKF(EKF):
    def __init__(self, dt):
        EKF.__init__(self, 3, 2, 2)
        self.dt = dt

    def predict_x(self, u):
        self.x = self.move(self.x, u, self.dt)


    def move(self, x, u, dt):
        h = x[2, 0]
        v = u[0]
        w = u[1]

        if w == 0:
            # approximate straight line with huge radius
            w = 1.e-30
        r = v/w # radius

        sinh = sin(h)
        sinhwdt = sin(h + w*dt)
        cosh = cos(h)
        coshwdt = cos(h + w*dt)

        return x + array([[-r*sinh + r*sinhwdt],
                          [r*cosh - r*coshwdt],
                          [w*dt]])


def H_of(x, landmark_pos):
    """ compute Jacobian of H matrix for state x """

    mx = landmark_pos[0]
    my = landmark_pos[1]
    q = (mx - x[0, 0])**2 + (my - x[1, 0])**2

    H = array(
            [[-(mx - x[0, 0]) / sqrt(q), -(my - x[1, 0]) / sqrt(q), 0],
             [ (my - x[1, 0]) / q,       -(mx - x[0, 0]) / q,      -1]])
    return H


def Hx(x, landmark_pos):
    """ takes a state variable and returns the measurement that would
    correspond to that state.
    """
    mx = landmark_pos[0]
    my = landmark_pos[1]
    q = (mx - x[0, 0])**2 + (my - x[1, 0])**2

    Hx = array([[sqrt(q)],
                [atan2(my - x[1, 0], mx - x[0, 0]) - x[2, 0]]])
    return Hx

dt = 1.0
ekf = RobotEKF(dt)

#np.random.seed(1234)

m = array([[5, 10],
           [10, 5],
           [15, 15]])

ekf.x = array([[2, 6, .3]]).T
u = array([.5, .01])
ekf.P = np.diag([1., 1., 1.])
ekf.R = np.diag([sigma_r**2, sigma_h**2])
c = [0, 1, 2]

xp = ekf.x.copy()

plt.scatter(m[:, 0], m[:, 1])
for i in range(300):
    xp = ekf.move(xp, u, dt/10.) # simulate robot
    plt.plot(xp[0], xp[1], ',', color='g')

    if i % 10 == 0:
        h = ekf.x[2, 0]
        v = u[0]
        w = u[1]

        if w == 0:
            # approximate straight line with huge radius
            w = 1.e-30
        r = v/w # radius

        sinh = sin(h)
        sinhwdt = sin(h + w*dt)
        cosh = cos(h)
        coshwdt = cos(h + w*dt)

        ekf.F = array(
           [[1, 0, -r*cosh + r*coshwdt],
            [0, 1, -r*sinh + r*sinhwdt],
            [0, 0, 1]])

        V = array(
            [[(-sinh + sinhwdt)/w, v*(sin(h)-sinhwdt)/(w**2) + v*coshwdt*dt/w],
             [(cosh - coshwdt)/w, -v*(cosh-coshwdt)/(w**2) + v*sinhwdt*dt/w],
             [0, dt]])

        # covariance of motion noise in control space
        M = array([[a1*v**2 + a2*w**2, 0],
                   [0, a3*v**2 + a4*w**2]])

        ekf.Q = dot(V, M).dot(V.T)

        ekf.predict(u=u)

        for lmark in m:
            d = sqrt((lmark[0] - xp[0, 0])**2 + (lmark[1] - xp[1, 0])**2)  + randn()*sigma_r
            a = atan2(lmark[1] - xp[1, 0], lmark[0] - xp[0, 0]) - xp[2, 0] + randn()*sigma_h
            z = np.array([[d], [a]])

            ekf.update(z, HJacobian=H_of, Hx=Hx, residual=residual,
                       args=(lmark), hx_args=(lmark))

        plot_covariance_ellipse((ekf.x[0,0], ekf.x[1,0]), ekf.P[0:2, 0:2], std=10,
                                facecolor='g', alpha=0.3)

    #plt.plot(ekf.x[0], ekf.x[1], 'x', color='r')

plt.axis('equal')
plt.show()

