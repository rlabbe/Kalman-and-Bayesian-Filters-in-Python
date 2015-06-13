# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:13:23 2015

@author: rlabbe
"""

from filterpy.common import plot_covariance_ellipse
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from math import tan, sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
from numpy.random import randn, seed



def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi]
        x -= 2 * np.pi
    return x


def residual_h(aa, bb):
    y = aa - bb
    for i in range(0, len(y), 2):
        y[i + 1] = normalize_angle(y[i + 1])
    return y


def residual_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y


def move(x, u, dt, wheelbase):
    h = x[2]
    v = u[0]
    steering_angle = u[1]

    dist = v*dt

    if abs(steering_angle) > 0.001:
        b = dist / wheelbase * tan(steering_angle)
        r = wheelbase / tan(steering_angle) # radius

        sinh = sin(h)
        sinhb = sin(h + b)
        cosh = cos(h)
        coshb = cos(h + b)

        return x + array([-r*sinh + r*sinhb, r*cosh - r*coshb, b])
    else:
        return x + array([dist*cos(h), dist*sin(h), 0])


def state_mean(sigmas, Wm):
    x = np.zeros(3)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))

    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = atan2(sum_sin, sum_cos)

    return x


def z_mean(sigmas, Wm):
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)

    for z in range(0, z_count, 2):
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), Wm))

        x[z] = np.sum(np.dot(sigmas[:,z], Wm))
        x[z+1] = atan2(sum_sin, sum_cos)
    return x


def fx(x, dt, u):
    return move(x, u, dt, wheelbase)


def Hx(x, landmark):
    """ takes a state variable and returns the measurement that would
    correspond to that state.
    """

    hx = []
    for lmark in landmark:
        px, py = lmark
        dist = sqrt((px - x[0])**2 + (py - x[1])**2)
        angle = atan2(py - x[1], px - x[0])
        hx.extend([dist, normalize_angle(angle - x[2])])
    return np.array(hx)


m = array([[5., 10], [10, 5], [15, 15], [20., 16], [0, 30], [50, 30], [40, 10]])
#m = array([[5, 10], [10, 5], [15, 15], [20, 5],[5,5], [8, 8.4]])#, [0, 30], [50, 30], [40, 10]])
#m = array([[5, 10], [10, 5]])#, [0, 30], [50, 30], [40, 10]])
#m = array([[5., 10], [10, 5]])
#m = array([[5., 10], [10, 5]])


sigma_r = .3
sigma_h =  .1#radians(.5)#np.radians(1)
#sigma_steer =  radians(10)
dt = 0.1
wheelbase = 0.5

points = MerweScaledSigmaPoints(n=3, alpha=.1, beta=2, kappa=0, subtract=residual_x)
#points = JulierSigmaPoints(n=3,  kappa=3)
ukf= UKF(dim_x=3, dim_z=2*len(m), fx=fx, hx=Hx, dt=dt, points=points,
         x_mean_fn=state_mean, z_mean_fn=z_mean,
         residual_x=residual_x, residual_z=residual_h)
ukf.x = array([2, 6, .3])
ukf.P = np.diag([.1, .1, .05])
ukf.R = np.diag([sigma_r**2, sigma_h**2]* len(m))
ukf.Q =np.eye(3)*.00001


u = array([1.1, 0.])

xp = ukf.x.copy()


plt.cla()
plt.scatter(m[:, 0], m[:, 1])

cmds = [[v, .0] for v in np.linspace(0.001, 1.1, 30)]
cmds.extend([cmds[-1]]*50)

v = cmds[-1][0]
cmds.extend([[v, a] for a in np.linspace(0, np.radians(2), 15)])
cmds.extend([cmds[-1]]*100)

cmds.extend([[v, a] for a in np.linspace(np.radians(2), -np.radians(2), 15)])
cmds.extend([cmds[-1]]*200)

cmds.extend([[v, a] for a in np.linspace(-np.radians(2), 0, 15)])
cmds.extend([cmds[-1]]*150)


seed(12)
cmds = np.array(cmds)

cindex = 0
u = cmds[0]

track = []

std = 16
while cindex < len(cmds):
    u = cmds[cindex]
    xp = move(xp, u, dt, wheelbase) # simulate robot
    track.append(xp)

    ukf.predict(fx_args=u)

    if cindex % 20 == 0:
        plot_covariance_ellipse((ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=std,
                                facecolor='b', alpha=0.58)

    #print(cindex, ukf.P.diagonal())
    #print(ukf.P.diagonal())
    z = []
    for lmark in m:
        d = sqrt((lmark[0] - xp[0])**2 + (lmark[1] - xp[1])**2) + randn()*sigma_r
        bearing = atan2(lmark[1] - xp[1], lmark[0] - xp[0])
        a = normalize_angle(bearing - xp[2] + randn()*sigma_h)
        z.extend([d, a])

        #if cindex % 20 == 0:
        #    plt.plot([lmark[0], lmark[0] - d*cos(a+xp[2])], [lmark[1], lmark[1]-d*sin(a+xp[2])], color='r')

        if cindex  == 1197:
            plt.plot([lmark[0], lmark[0] - d2*cos(a2+xp[2])],
                     [lmark[1], lmark[1] - d2*sin(a2+xp[2])], color='r')
            plt.plot([lmark[0], lmark[0] - d*cos(a+xp[2])],
                     [lmark[1], lmark[1] - d*sin(a+xp[2])], color='k')

    ukf.update(np.array(z), hx_args=(m,))

    if cindex % 20 == 0:
        plot_covariance_ellipse((ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=std,
                                 facecolor='g', alpha=0.5)
    cindex += 1


track = np.array(track)
plt.plot(track[:, 0], track[:,1], color='k')

plt.axis('equal')
plt.title("UKF Robot localization")
plt.show()
