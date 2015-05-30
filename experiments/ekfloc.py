# -*- coding: utf-8 -*-
"""
Created on Sun May 24 08:39:36 2015

@author: Roger
"""

#x = x x' y y' theta

from math import cos, sin, sqrt, atan2
import numpy as np
from numpy import array, dot
from numpy.linalg import pinv


def print_x(x):
    print(x[0, 0], x[1, 0], np.degrees(x[2, 0]))


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


def ekfloc_predict(x, P, u, dt):

    h = x[2]
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

    G = array(
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



    x = x + array([[-r*sinh + r*sinhwdt],
                   [r*cosh - r*coshwdt],
                   [w*dt]])

    P = dot(G, P).dot(G.T) + dot(V, M).dot(V.T)

    return x, P

def ekfloc(x, P, u, zs, c, m, dt):

    h = x[2]
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

    F = array(
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


    x = x + array([[-r*sinh + r*sinhwdt],
                   [r*cosh - r*coshwdt],
                   [w*dt]])

    P = dot(F, P).dot(F.T) + dot(V, M).dot(V.T)


    R = np.diag([sigma_r**2, sigma_h**2, sigma_s**2])

    for i, z in enumerate(zs):
        j = c[i]

        q = (m[j][0] - x[0, 0])**2 + (m[j][1] - x[1, 0])**2

        z_est = array([sqrt(q),
                       atan2(m[j][1] - x[1, 0], m[j][0] - x[0, 0]) - x[2, 0],
                       0])


        H = array(
            [[-(m[j, 0] - x[0, 0]) / sqrt(q), -(m[j, 1] - x[1, 0]) / sqrt(q), 0],
             [ (m[j, 1] - x[1, 0]) / q,       -(m[j, 0] - x[0, 0]) / q,      -1],
             [0,                              0,                              0]])



        S = dot(H, P).dot(H.T) + R

        #print('S', S)
        K = dot(P, H.T).dot(pinv(S))
        y = z - z_est
        normalize_angle(y, 1)
        y = array([y]).T
        #print('y', y)

        x = x + dot(K, y)
        I = np.eye(P.shape[0])
        I_KH = I - dot(K, H)
        #print('i', I_KH)

        P = dot(I_KH, P).dot(I_KH.T) + dot(K, R).dot(K.T)

    return x, P



def ekfloc2(x, P, u, zs, c, m, dt):

    h = x[2]
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

    F = array(
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


    x = x + array([[-r*sinh + r*sinhwdt],
                   [r*cosh - r*coshwdt],
                   [w*dt]])


    P = dot(F, P).dot(F.T) + dot(V, M).dot(V.T)



    R = np.diag([sigma_r**2, sigma_h**2])

    for i, z in enumerate(zs):
        j = c[i]

        q = (m[j][0] - x[0, 0])**2 + (m[j][1] - x[1, 0])**2

        z_est = array([sqrt(q),
                       atan2(m[j][1] - x[1, 0], m[j][0] - x[0, 0]) - x[2, 0]])

        H = array(
            [[-(m[j, 0] - x[0, 0]) / sqrt(q), -(m[j, 1] - x[1, 0]) / sqrt(q), 0],
             [ (m[j, 1] - x[1, 0]) / q,       -(m[j, 0] - x[0, 0]) / q,      -1]])


        S = dot(H, P).dot(H.T) + R

        #print('S', S)
        K = dot(P, H.T).dot(pinv(S))
        y = z - z_est
        normalize_angle(y, 1)
        y = array([y]).T
        #print('y', y)

        x = x + dot(K, y)
        print('x', x)
        I = np.eye(P.shape[0])
        I_KH = I - dot(K, H)

        P = dot(I_KH, P).dot(I_KH.T) + dot(K, R).dot(K.T)

    return x, P

m = array([[5, 5],
           [7,6],
           [4, 8]])

x = array([[2, 6, .3]]).T
u = array([.5, .01])
P = np.diag([1., 1., 1.])
c = [0, 1, 2]



import matplotlib.pyplot as plt
from numpy.random import randn
from filterpy.common import plot_covariance_ellipse
from filterpy.kalman import KalmanFilter
plt.figure()
plt.plot(m[:, 0], m[:, 1], 'o')
plt.plot(x[0], x[1], 'x', color='b', ms=20)

xp = x.copy()
dt = 0.1
np.random.seed(1234)

for i in range(1000):
    xp, _ = ekfloc_predict(xp, P, u, dt)
    plt.plot(xp[0], xp[1], 'x', color='g', ms=20)

    if i % 10 == 0:
        zs = []

        for lmark in m:
            d = sqrt((lmark[0] - xp[0, 0])**2 + (lmark[1] - xp[1, 0])**2)  + randn()*sigma_r
            a = atan2(lmark[1] - xp[1, 0], lmark[0] - xp[0, 0]) - xp[2, 0] + randn()*sigma_h
            zs.append(np.array([d, a]))

        x, P = ekfloc2(x, P, u, zs, c, m, dt*10)


        if P[0,0] < 10000:
            plot_covariance_ellipse((x[0,0], x[1,0]), P[0:2, 0:2], std=2,
                                    facecolor='g', alpha=0.3)

    plt.plot(x[0], x[1], 'x', color='r')

plt.axis('equal')
plt.show()
