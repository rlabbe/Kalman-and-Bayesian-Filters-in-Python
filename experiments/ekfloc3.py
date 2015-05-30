# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:23:57 2015

@author: rlabbe
"""


from math import cos, sin, sqrt, atan2, tan
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


sigma_r = 0.1
sigma_h =  np.radians(1)
sigma_steer =  np.radians(1)

#only partway through. predict is using old control and movement code. computation of m uses
#old u.

class RobotEKF(EKF):
    def __init__(self, dt, wheelbase):
        EKF.__init__(self, 3, 2, 2)
        self.dt = dt
        self.wheelbase = wheelbase

    def predict(self, u=0):

        self.x = self.move(self.x, u, self.dt)

        h = self.x[2, 0]
        v = u[0]
        steering_angle = u[1]

        dist = v*self.dt

        if abs(steering_angle) < 0.0001:
            # approximate straight line with huge radius
            r = 1.e-30
        b = dist / self.wheelbase * tan(steering_angle)
        r = self.wheelbase / tan(steering_angle) # radius


        sinh = sin(h)
        sinhb = sin(h + b)
        cosh = cos(h)
        coshb = cos(h + b)

        F = array([[1., 0., -r*cosh + r*coshb],
                   [0., 1., -r*sinh + r*sinhb],
                   [0., 0., 1.]])

        w = self.wheelbase

        F = array([[1., 0., (-w*cosh + w*coshb)/tan(steering_angle)],
                   [0., 1., (-w*sinh + w*sinhb)/tan(steering_angle)],
                   [0., 0., 1.]])

        print('F', F)

        V = array(
            [[-r*sinh + r*sinhb, 0],
             [r*cosh + r*coshb, 0],
             [0, 0]])


        t2 = tan(steering_angle)**2
        V = array([[0, w*sinh*(-t2-1)/t2 + w*sinhb*(-t2-1)/t2],
                   [0, w*cosh*(-t2-1)/t2 - w*coshb*(-t2-1)/t2],
                   [0,0]])


        t2 = tan(steering_angle)**2

        a = steering_angle
        d = v*dt
        it = dt*v*tan(a)/w + h



        V[0,0] = dt*cos(d/w*tan(a) + h)
        V[0,1] = (dt*v*(t2+1)*cos(it)/tan(a) -
                  w*sinh*(-t2-1)/t2 +
                  w*(-t2-1)*sin(it)/t2)


        print(dt*v*(t2+1)*cos(it)/tan(a))
        print(w*sinh*(-t2-1)/t2)
        print(w*(-t2-1)*sin(it)/t2)



        V[1,0] = dt*sin(it)

        V[1,1] = (d*(t2+1)*sin(it)/tan(a) + w*cosh/t2*(-t2-1) -
                  w*(-t2-1)*cos(it)/t2)

        V[2,0] = dt/w*tan(a)
        V[2,1] = d/w*(t2+1)

        theta = h

        v01 = (dt*v*(tan(a)**2 + 1)*cos(dt*v*tan(a)/w + theta)/tan(a) -
               w*(-tan(a)**2 - 1)*sin(theta)/(tan(a)**2) +
               w*(-tan(a)**2 - 1)*sin(dt*v*tan(a)/w + theta)/(tan(a)**2))

        print(dt*v*(tan(a)**2 + 1)*cos(dt*v*tan(a)/w + theta)/tan(a))
        print(w*(-tan(a)**2 - 1)*sin(theta)/(tan(a)**2))
        print(w*(-tan(a)**2 - 1)*sin(dt*v*tan(a)/w + theta)/(tan(a)**2))

        '''v11 = (dt*v*(tan(a)**2 + 1)*sin(dt*v*tan(a)/w + theta)/tan(a) +
               w*(-tan(a)**2 - 1)*cos(theta)/tan(a)**2 -
               w*(-tan(a)**2 - 1)*cos(dt*v*tan(a)/w + theta)/tan(a)**2)


        print(dt*v*(tan(a)**2 + 1)*sin(dt*v*tan(a)/w + theta)/tan(a))
        print(w*(-tan(a)**2 - 1)*cos(theta)/tan(a)**2)
        print(w*(-tan(a)**2 - 1)*cos(dt*v*tan(a)/w + theta)/tan(a)**2)
        print(v11)
        print(V[1,1])
        1/0

        V[1,1] = v11'''

        print(V)


        # covariance of motion noise in control space
        M = array([[0.1*v**2, 0],
                   [0,         sigma_steer**2]])


        fpf = dot(F, self.P).dot(F.T)
        Q =  dot(V, M).dot(V.T)
        print('FPF', fpf)
        print('V', V)
        print('Q', Q)
        print()
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
    bearing to a landmark for state x """

    px = p[0]
    py = p[1]
    hyp = (px - x[0, 0])**2 + (py - x[1, 0])**2
    dist = np.sqrt(hyp)

    H = array(
        [[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
         [ (py - x[1, 0]) / hyp,  -(px - x[0, 0]) / hyp, -1]])
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

        ekf.predict(u=u)

        plot_covariance_ellipse((ekf.x[0,0], ekf.x[1,0]), ekf.P[0:2, 0:2], std=2,
                                facecolor='b', alpha=0.3)

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

