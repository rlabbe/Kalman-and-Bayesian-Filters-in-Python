# -*- coding: utf-8 -*-
"""
Created on Sat Jul 05 09:54:39 2014

@author: rlabbe
"""

from __future__ import division
import matplotlib.pyplot as plt
from scipy.integrate import ode


class BallEuler(object):
    def __init__(self, y=100., vel=10.):
        self.x = 0.
        self.y = y
        self.vel = vel
        self.y_vel = 0.0



    def step (self, dt):

        g = -9.8


        self.x += self.vel*dt
        self.y += self.y_vel*dt

        self.y_vel += g*dt

        #print self.x, self.y



def rk4(y, x, dx, f):
    """computes 4th order Runge-Kutta for dy/dx.
    y is the initial value for y
    x is the initial value for x
    dx is the difference in x (e.g. the time step)
    f is a callable function (y, x) that you supply to compute dy/dx for
      the specified values.
      
    
    """
    
    k1 = dx * f(y, x)
    k2 = dx * f(y + 0.5*k1, x + 0.5*dx)
    k3 = dx * f(y + 0.5*k2, x + 0.5*dx)
    k4 = dx * f(y + k3, x + dx)
    
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def fy(y, t):
    """ returns velocity of ball at time t. 
    return -9.8*t


def fx(y, t):
    """ returns velocity of ball. Need to set vx.vel prior to first call
    return fx.vel
        
class BallRungeKutta(object):
    def __init__(self, y=100., vel=10.):
        self.x = 0.
        self.y = y
        self.vel = vel
        self.y_vel = 0.0
        self.t = 0
        
        fx.vel = vel


    def step2 (self, dt):
        self.x = rk4 (self.x, self.t, dt, fx)
        self.y = rk4 (self.y, self.t, dt, fy)
        self.t += dt

        print self.x, self.y


if __name__ == "__main__":

    dt = 1./30
    y0 = 15.
    vel = 100.
    be = BallEuler (y=y0, vel=vel)
    brk = BallRungeKutta (y=y0, vel=vel)


    solver = ode(f).set_integrator('dopri5')
    solver.set_initial_value(y0, 0)
    t = 0
    y = y0

    
    while be.y >= 0:
        be.step (dt)
        #plt.scatter (be.x, be.y, color='red')


    while brk.y >= 0:
        brk.step2 (dt)

        y = solver.integrate(t+dt)
        t += dt
        #print brk.x, y[0]

        plt.scatter (brk.x, brk.y, color='blue', marker='v')
        plt.scatter (brk.x, y[0], color='green', marker='+')
