# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 09:55:24 2015

@author: rlabbe
"""

from math import radians, sin, cos, sqrt, exp, atan2, radians
from numpy import array, asarray
from numpy.random import randn
import numpy as np
import math
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import runge_kutta4







class BaseballPath(object):
    def __init__(self, x0, y0, launch_angle_deg, velocity_ms,
                 noise=(1.0,1.0)):
        """ Create 2D baseball path object
           (x = distance from start point in ground plane,
            y=height above ground)

        x0,y0            initial position
        launch_angle_deg angle ball is travelling respective to
                         ground plane
        velocity_ms      speeed of ball in meters/second
        noise            amount of noise to add to each position
                         in (x,y)
        """

        omega = radians(launch_angle_deg)
        self.v_x = velocity_ms * cos(omega)
        self.v_y = velocity_ms * sin(omega)

        self.x = x0
        self.y = y0

        self.noise = noise


    def drag_force(self, velocity):
        """ Returns the force on a baseball due to air drag at
        the specified velocity. Units are SI
        """
        B_m = 0.0039 + 0.0058 / (1. + exp((velocity-35.)/5.))
        return B_m * velocity


    def update(self, dt, vel_wind=0.):
        """ compute the ball position based on the specified time
        step and wind velocity. Returns (x,y) position tuple
        """

        # Euler equations for x and y
        self.x += self.v_x*dt
        self.y += self.v_y*dt

        # force due to air drag
        v_x_wind = self.v_x - vel_wind
        v = sqrt(v_x_wind**2 + self.v_y**2)
        F = self.drag_force(v)

        # Euler's equations for velocity
        self.v_x = self.v_x - F*v_x_wind*dt
        self.v_y = self.v_y - 9.81*dt - F*self.v_y*dt

        return (self.x, self.y)



radar_pos = (100,0)
omega = 45.


def radar_sense(baseball, noise_rng, noise_brg):
    x, y = baseball.x, baseball.y

    rx, ry = radar_pos[0], radar_pos[1]

    rng = ((x-rx)**2 + (y-ry)**2) ** .5
    bearing = atan2(y-ry, x-rx)

    rng += randn() * noise_rng
    bearing += radians(randn() * noise_brg)

    return (rng, bearing)


ball = BaseballPath(x0=0, y0=1, launch_angle_deg=45,
                    velocity_ms=60, noise=[0,0])


'''
xs = []
ys = []
dt = 0.05
y = 1

while y > 0:
    x,y = ball.update(dt)
    xs.append(x)
    ys.append(y)

plt.plot(xs, ys)
plt.axis('equal')


plt.show()

'''


dt = 1/30.


def hx(x):
    global radar_pos

    dx = radar_pos[0] - x[0]
    dy = radar_pos[1] - x[2]
    rng = (dx*dx + dy*dy)**.5
    bearing = atan2(-dy, -dx)

    #print(x)
    #print('hx:', rng, np.degrees(bearing))

    return array([rng, bearing])




def fx(x, dt):

    fx.ball.x = x[0]
    fx.ball.y = x[2]
    fx.ball.vx = x[1]
    fx.ball.vy = x[3]

    N = 10
    ball_dt = dt/float(N)

    for i in range(N):
        fx.ball.update(ball_dt)

    #print('fx', fx.ball.x, fx.ball.v_x, fx.ball.y, fx.ball.v_y)

    return array([fx.ball.x, fx.ball.v_x, fx.ball.y, fx.ball.v_y])


fx.ball = BaseballPath(x0=0, y0=1, launch_angle_deg=45,
                       velocity_ms=60, noise=[0,0])


y = 1.
x = 0.
theta = 35. # launch angle
v0 = 50.


ball = BaseballPath(x0=x, y0=y, launch_angle_deg=theta,
                    velocity_ms=v0, noise=[.3,.3])

kf = UKF(dim_x=4, dim_z=2, dt=dt, hx=hx, fx=fx, kappa=0)

#kf.R *= r

kf.R[0,0] = 0.1
kf.R[1,1] = radians(0.2)
omega = radians(omega)
vx = cos(omega) * v0
vy = sin(omega) * v0

kf.x = array([x, vx, y, vy])
kf.R*= 0.01
#kf.R[1,1] = 0.01
kf.P *= 10

f1 = kf


t = 0
xs = []
ys = []

while y > 0:
    t += dt
    x,y = ball.update(dt)
    z = radar_sense(ball, 0, 0)
    #print('z', z)
    #print('ball', ball.x, ball.v_x, ball.y, ball.v_y)


    f1.predict()
    f1.update(z)
    xs.append(f1.x[0])
    ys.append(f1.x[2])
    f1.predict()

    p1 = plt.scatter(x, y, color='r', marker='o', s=75, alpha=0.5)

p2, = plt.plot (xs, ys, lw=2, marker='o')
#p3, = plt.plot (xs2, ys2, lw=4)
#plt.legend([p1,p2, p3],
#           ['Measurements', 'Kalman filter(R=0.5)', 'Kalman filter(R=10)'],
#           loc='best', scatterpoints=1)
plt.show()

