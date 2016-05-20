# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.


Code supporting the book

Kalman and Bayesian Filters in Python
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


This is licensed under an MIT license. See the LICENSE.txt file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import code.book_plots as bp
import filterpy.kalman as kf
from math import radians, sin, cos, sqrt, exp
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random

def ball_kf(x, y, omega, v0, dt, r=0.5, q=0.02):

    g = 9.8 # gravitational constant
    f1 = kf.KalmanFilter(dim_x=5, dim_z=2)

    ay = .5*dt**2
    f1.F = np.array ([[1, dt,  0,  0,  0],   # x   = x0+dx*dt
                      [0,  1,  0,  0,  0],   # dx  = dx
                      [0,  0,  1, dt, ay],   # y   = y0 +dy*dt+1/2*g*dt^2
                      [0,  0,  0,  1, dt],   # dy  = dy0 + ddy*dt
                      [0,  0,  0,  0, 1]])   # ddy = -g.

    f1.H = np.array([
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0]])

    f1.R *= r
    f1.Q *= q

    omega = radians(omega)
    vx = cos(omega) * v0
    vy = sin(omega) * v0

    f1.x = np.array([[x,vx,y,vy,-9.8]]).T

    return f1


def plot_radar(xs, track, time):
    plt.figure()
    bp.plot_track(time, track[:, 0])
    bp.plot_filter(time, xs[:, 0])
    plt.legend(loc=4)
    plt.xlabel('time (sec)')
    plt.ylabel('position (m)')

    plt.figure()
    bp.plot_track(time, track[:, 1])
    bp.plot_filter(time, xs[:, 1])
    plt.legend(loc=4)
    plt.xlabel('time (sec)')
    plt.ylabel('velocity (m/s)')

    plt.figure()
    bp.plot_track(time, track[:, 2])
    bp.plot_filter(time, xs[:, 2])
    plt.ylabel('altitude (m)')
    plt.legend(loc=4)
    plt.xlabel('time (sec)')
    plt.ylim((900, 1600))
    plt.show()


def plot_bicycle():
    plt.clf()
    plt.axes()
    ax = plt.gca()
    #ax.add_patch(plt.Rectangle((10,0), 10, 20,  fill=False, ec='k')) #car
    ax.add_patch(plt.Rectangle((21,1), .75, 2,  fill=False, ec='k')) #wheel
    ax.add_patch(plt.Rectangle((21.33,10), .75, 2,  fill=False, ec='k', angle=20)) #wheel
    ax.add_patch(plt.Rectangle((21.,4.), .75, 2,  fill=True, ec='k', angle=5, ls='dashdot', alpha=0.3)) #wheel

    plt.arrow(0, 2,  20.5, 0,  fc='k', ec='k', head_width=0.5, head_length=0.5)
    plt.arrow(0, 2,  20.4, 3,  fc='k', ec='k', head_width=0.5, head_length=0.5)
    plt.arrow(21.375, 2., 0, 8.5,  fc='k', ec='k', head_width=0.5, head_length=0.5)
    plt.arrow(23, 2,  0, 2.5,  fc='k', ec='k', head_width=0.5, head_length=0.5)

    #ax.add_patch(plt.Rectangle((10,0), 10, 20,  fill=False, ec='k'))
    plt.text(11, 1.0, "R", fontsize=18)
    plt.text(8, 2.2, r"$\beta$", fontsize=18)
    plt.text(20.4, 13.5, r"$\alpha$", fontsize=18)
    plt.text(21.6, 7, "w (wheelbase)", fontsize=18)
    plt.text(0, 1, "C", fontsize=18)
    plt.text(24, 3, "d", fontsize=18)
    plt.plot([21.375, 21.25], [11, 14], color='k', lw=1)
    plt.plot([21.375, 20.2], [11, 14], color='k', lw=1)
    plt.axis('scaled')
    plt.xlim(0,25)
    plt.axis('off')
    plt.show()

#plot_bicycle()

class BaseballPath(object):
    def __init__(self, x0, y0, launch_angle_deg, velocity_ms, noise=(1.0,1.0)):
        """ Create baseball path object in 2D (y=height above ground)

        x0,y0 initial position
        launch_angle_deg angle ball is travelling respective to ground plane
        velocity_ms speeed of ball in meters/second
        noise amount of noise to add to each reported position in (x,y)
        """

        omega = radians(launch_angle_deg)
        self.v_x = velocity_ms * cos(omega)
        self.v_y = velocity_ms * sin(omega)

        self.x = x0
        self.y = y0

        self.noise = noise


    def drag_force (self, velocity):
        """ Returns the force on a baseball due to air drag at
        the specified velocity. Units are SI
        """
        B_m = 0.0039 + 0.0058 / (1. + exp((velocity-35.)/5.))
        return B_m * velocity


    def update(self, dt, vel_wind=0.):
        """ compute the ball position based on the specified time step and
        wind velocity. Returns (x,y) position tuple.
        """

        # Euler equations for x and y
        self.x += self.v_x*dt
        self.y += self.v_y*dt

        # force due to air drag
        v_x_wind = self.v_x - vel_wind

        v = sqrt (v_x_wind**2 + self.v_y**2)
        F = self.drag_force(v)

        # Euler's equations for velocity
        self.v_x = self.v_x - F*v_x_wind*dt
        self.v_y = self.v_y - 9.81*dt - F*self.v_y*dt

        return (self.x + random.randn()*self.noise[0],
                self.y + random.randn()*self.noise[1])



def plot_ball():
    y = 1.
    x = 0.
    theta = 35. # launch angle
    v0 = 50.
    dt = 1/10.   # time step

    ball = BaseballPath(x0=x, y0=y, launch_angle_deg=theta, velocity_ms=v0, noise=[.3,.3])
    f1 = ball_kf(x,y,theta,v0,dt,r=1.)
    f2 = ball_kf(x,y,theta,v0,dt,r=10.)
    t = 0
    xs = []
    ys = []
    xs2 = []
    ys2 = []

    while f1.x[2,0] > 0:
        t += dt
        x,y = ball.update(dt)
        z = np.mat([[x,y]]).T

        f1.update(z)
        f2.update(z)
        xs.append(f1.x[0,0])
        ys.append(f1.x[2,0])
        xs2.append(f2.x[0,0])
        ys2.append(f2.x[2,0])
        f1.predict()
        f2.predict()

        p1 = plt.scatter(x, y, color='green', marker='o', s=75, alpha=0.5)

    p2, = plt.plot (xs, ys,lw=2)
    p3, = plt.plot (xs2, ys2,lw=4, c='r')
    plt.legend([p1,p2, p3], ['Measurements', 'Kalman filter(R=0.5)', 'Kalman filter(R=10)'])
    plt.show()


def show_radar_chart():
    plt.xlim([0.9,2.5])
    plt.ylim([0.5,2.5])

    plt.scatter ([1,2],[1,2])
    #plt.scatter ([2],[1],marker='o')
    ax = plt.axes()

    ax.annotate('', xy=(2,2), xytext=(1,1),
                arrowprops=dict(arrowstyle='->', ec='r',shrinkA=3, shrinkB=4))

    ax.annotate('', xy=(2,1), xytext=(1,1),
                arrowprops=dict(arrowstyle='->', ec='b',shrinkA=0, shrinkB=0))

    ax.annotate('', xy=(2,2), xytext=(2,1),
                arrowprops=dict(arrowstyle='->', ec='b',shrinkA=0, shrinkB=4))



    ax.annotate('$\epsilon$', xy=(1.2, 1.05), color='b')
    ax.annotate('Aircraft', xy=(2.04,2.), color='b')
    ax.annotate('altitude (y)', xy=(2.04,1.5), color='k')
    ax.annotate('x', xy=(1.5, .9))
    ax.annotate('Radar', xy=(.95, 0.8))
    ax.annotate('Slant\n  (r)', xy=(1.5,1.62), color='r')

    plt.title("Radar Tracking")
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.show()


def show_linearization():
    xs =  np.arange(0, 2, 0.01)
    ys = [x**2 - 2*x for x in xs]

    def y(x):
        return x - 2.25

    plt.plot(xs, ys, label='$f(x)=x^2−2x$')
    plt.plot([1, 2], [y(1), y(2)], color='k', ls='--', label='linearization')
    plt.axes().axvline(1.5, lw=1, c='k')
    plt.xlim(0, 2)
    plt.ylim([-1.5, 0.0])
    plt.title('Linearization of $f(x)$ at $x=1.5$')
    plt.xlabel('$x=1.5$')
    plt.legend(loc=4)
    plt.show()