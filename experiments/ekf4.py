# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:24:01 2015

@author: rlabbe
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:23:57 2015

@author: rlabbe
"""


from math import cos, sin, sqrt, atan2, tan
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, dot
from numpy.random import randn
from filterpy.common import plot_covariance_ellipse
from filterpy.kalman import ExtendedKalmanFilter as EKF
from sympy import Matrix, symbols
import sympy

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


sigma_r = 1
sigma_h =  .1#np.radians(1)
sigma_steer =  np.radians(1)


class RobotEKF(EKF):
    def __init__(self, dt, wheelbase):
        EKF.__init__(self, 3, 2, 2)
        self.dt = dt
        self.wheelbase = wheelbase

        a, x, y, v, w, theta, time = symbols(
            'a, x, y, v, w, theta, t')

        d = v*time
        beta = (d/w)*sympy.tan(a)
        r = w/sympy.tan(a)


        self.fxu = Matrix([[x-r*sympy.sin(theta)+r*sympy.sin(theta+beta)],
                           [y+r*sympy.cos(theta)-r*sympy.cos(theta+beta)],
                           [theta+beta]])

        self.F_j = self.fxu.jacobian(Matrix([x, y, theta]))
        self.V_j = self.fxu.jacobian(Matrix([v, a]))

        self.subs = {x: 0, y: 0, v:0, a:0, time:dt, w:wheelbase, theta:0}
        self.x_x = x
        self.x_y = y
        self.v = v
        self.a = a
        self.theta = theta


    def predict(self, u=0):


        self.x = self.move(self.x, u, self.dt)

        self.subs[self.theta] = self.x[2, 0]
        self.subs[self.v] = u[0]
        self.subs[self.a] = u[1]


        F = array(self.F_j.evalf(subs=self.subs)).astype(float)
        V = array(self.V_j.evalf(subs=self.subs)).astype(float)

        # covariance of motion noise in control space
        M = array([[0.1*u[0]**2, 0],
                   [0,         sigma_steer**2]])

        self.P = dot(F, self.P).dot(F.T) + dot(V, M).dot(V.T)


    def move(self, x, u, dt):
        h = x[2, 0]
        v = u[0]
        steering_angle = u[1]

        dist = v*dt

        if abs(steering_angle) < 0.0001:
            # approximate straight line with huge radius
            r = 1.e-30
        b = dist / self.wheelbase * tan(steering_angle)
        r = self.wheelbase / tan(steering_angle) # radius


        sinh = sin(h)
        sinhb = sin(h + b)
        cosh = cos(h)
        coshb = cos(h + b)

        return x + array([[-r*sinh + r*sinhb],
                          [r*cosh - r*coshb],
                          [b]])


def H_of(x, p):
    """ compute Jacobian of H matrix where h(x) computes the range and
    bearing to a landmark 'p' for state x """

    px = p[0]
    py = p[1]
    hyp = (px - x[0, 0])**2 + (py - x[1, 0])**2
    dist = np.sqrt(hyp)

    H = array(
        [[(-px + x[0, 0]) / dist, (-py + x[1, 0]) / dist, 0.],
         [ -(-py + x[1, 0]) / hyp,  -(px - x[0, 0]) / hyp, -1.]])
    return H


def Hx(x, p):
    """ takes a state variable and returns the measurement that would
    correspond to that state.
    """
    px = p[0]
    py = p[1]
    dist = np.sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)

    Hx = array([[dist],
                [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])
    return Hx

dt = 1.0
ekf = RobotEKF(dt, wheelbase=0.5)

#np.random.seed(1234)

m = array([[5, 10],
           [10, 5],
           [15, 15]])

ekf.x = array([[2, 6, .3]]).T
ekf.P = np.diag([.1, .1, .1])
ekf.R = np.diag([sigma_r**2, sigma_h**2])

u = array([1.1, .01])

xp = ekf.x.copy()

plt.figure()
plt.scatter(m[:, 0], m[:, 1])
for i in range(250):
    xp = ekf.move(xp, u, dt/10.) # simulate robot
    plt.plot(xp[0], xp[1], ',', color='g')

    if i % 10 == 0:

        ekf.predict(u=u)

        plot_covariance_ellipse((ekf.x[0,0], ekf.x[1,0]), ekf.P[0:2, 0:2], std=3,
                                facecolor='b', alpha=0.08)

        for lmark in m:
            d = sqrt((lmark[0] - xp[0, 0])**2 + (lmark[1] - xp[1, 0])**2)  + randn()*sigma_r
            a = atan2(lmark[1] - xp[1, 0], lmark[0] - xp[0, 0]) - xp[2, 0] + randn()*sigma_h
            z = np.array([[d], [a]])

            ekf.update(z, HJacobian=H_of, Hx=Hx, residual=residual,
                       args=(lmark), hx_args=(lmark))

        plot_covariance_ellipse((ekf.x[0,0], ekf.x[1,0]), ekf.P[0:2, 0:2], std=3,
                                facecolor='g', alpha=0.4)

    #plt.plot(ekf.x[0], ekf.x[1], 'x', color='r')

plt.axis('equal')
plt.title("EKF Robot localization")
plt.show()

