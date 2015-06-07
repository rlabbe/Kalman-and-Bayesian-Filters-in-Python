# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:13:23 2015

@author: rlabbe
"""

from filterpy.common import plot_covariance_ellipse
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from math import tan, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
from numpy.random import randn



def normalize_angle(x, index):
    def normalize(x):
        if x > np.pi:
            x -= 2*np.pi
        if x < -np.pi:
            x = 2*np.pi
        return x

    if x.ndim > 1:
        for i in range(len(x)):
            x[i, index] = normalize(x[i, index])

    else:
        x[index] = normalize(x[index])


def residual(a,b , index=1):
    y = a - b
    normalize_angle(y, index)
    return y


def residual_h(a, b):
    return residual(a, b, 1)

def residual_x(a, b):
    return residual(a, b, 2)


def move(x, u, dt, wheelbase):
    h = x[2]
    v = u[0]
    steering_angle = u[1]

    dist = v*dt

    if abs(steering_angle) < 0.0001:
        # approximate straight line with huge radius
        r = 1.e-30
    b = dist / wheelbase * tan(steering_angle)
    r = wheelbase / tan(steering_angle) # radius


    sinh = sin(h)
    sinhb = sin(h + b)
    cosh = cos(h)
    coshb = cos(h + b)

    return x + array([-r*sinh + r*sinhb, r*cosh - r*coshb, b])





def state_unscented_transform(Sigmas, Wm, Wc, noise_cov):
    """ Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.
    """

    kmax, n = Sigmas.shape

    x = np.zeros(3)
    sum_sin, sum_cos = 0., 0.

    for i in range(len(Sigmas)):
        s = Sigmas[i]
        x[0] += s[0] * Wm[i]
        x[1] += s[1] * Wm[i]
        sum_sin += sin(s[2])*Wm[i]
        sum_cos += cos(s[2])*Wm[i]

    x[2] = atan2(sum_sin, sum_cos)


    # new covariance is the sum of the outer product of the residuals
    # times the weights
    P = np. zeros((n, n))
    for k in range(kmax):
        y = residual_x(Sigmas[k],  x)
        P += Wc[k] * np.outer(y, y)

    if noise_cov is not None:
        P += noise_cov

    return (x, P)



def z_unscented_transform(Sigmas, Wm, Wc, noise_cov):
    """ Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.
    """

    kmax, n = Sigmas.shape

    x = np.zeros(2)
    sum_sin, sum_cos = 0., 0.

    for i in range(len(Sigmas)):
        s = Sigmas[i]
        x[0] += s[0] * Wm[i]
        sum_sin += sin(s[1])*Wm[i]
        sum_cos += cos(s[1])*Wm[i]

    x[1] = atan2(sum_sin, sum_cos)


    # new covariance is the sum of the outer product of the residuals
    # times the weights
    P = np.zeros((n, n))
    for k in range(kmax):
        y = residual_h(Sigmas[k], x)

        P += Wc[k] * np.outer(y, y)


    if noise_cov is not None:
        P += noise_cov

    return (x, P)


sigma_r = 1.
sigma_h =  .1#np.radians(1)
sigma_steer =  np.radians(.01)
dt = 1.0
wheelbase = 0.5

m = array([[5, 10],
           [10, 5],
           [15, 15]])


def fx(x, dt, u):
    return move(x, u, dt, wheelbase)


def Hx(x, landmark):
    """ takes a state variable and returns the measurement that would
    correspond to that state.
    """
    px = landmark[0]
    py = landmark[1]
    dist = np.sqrt((px - x[0])**2 + (py - x[1])**2)

    Hx = array([dist, atan2(py - x[1], px - x[0]) - x[2]])
    return Hx

points = MerweScaledSigmaPoints(n=3, alpha=1.e-3, beta=2, kappa=0)
ukf= UKF(dim_x=3, dim_z=2, fx=fx, hx=Hx, dt=dt, points=points)
ukf.x = array([2, 6, .3])
ukf.P = np.diag([.1, .1, .2])
ukf.R = np.diag([sigma_r**2, sigma_h**2])
ukf.Q = np.zeros((3,3))


u = array([1.1, .01])

xp = ukf.x.copy()

plt.figure()
plt.scatter(m[:, 0], m[:, 1])

for i in range(250):
    xp = move(xp, u, dt/10., wheelbase) # simulate robot
    plt.plot(xp[0], xp[1], ',', color='g')

    if i % 10 == 0:
        ukf.predict(fx_args=u, UT=state_unscented_transform)

        plot_covariance_ellipse((ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=3,
                                facecolor='b', alpha=0.08)

        for lmark in m:
            d = sqrt((lmark[0] - xp[0])**2 + (lmark[1] - xp[1])**2)  + randn()*sigma_r
            a = atan2(lmark[1] - xp[1], lmark[0] - xp[0]) - xp[2] + randn()*sigma_h
            z = np.array([d, a])

            ukf.update(z, hx_args=(lmark,), UT=z_unscented_transform,
                       residual_x=residual_x, residual_h=residual_h)

        plot_covariance_ellipse((ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=3,
                                facecolor='g', alpha=0.4)


    #plt.plot(ekf.x[0], ekf.x[1], 'x', color='r')

plt.axis('equal')
plt.title("UKF Robot localization")
plt.show()