# -*- coding: utf-8 -*-
"""
Created on Sat Jul 05 09:54:39 2014

@author: rlabbe
"""

from __future__ import division, print_function
import matplotlib.pyplot as plt
from scipy.integrate import ode
import math
import numpy as np
from numpy import random, radians, cos, sin


class BallTrajectory2D(object):
    def __init__(self, x0, y0, velocity, theta_deg=0., g=9.8, noise=[0.0,0.0]):
        theta = radians(theta_deg)
        self.vx0 = velocity * cos(theta)
        self.vy0 = velocity * sin(theta)
        
        self.x0 = x0
        self.y0 = y0
        self.x = x
        
        self.g = g
        self.noise = noise
        
    def position(self, t):
        """ returns (x,y) tuple of ball position at time t"""
        
        self.x = self.vx0*t + self.x0
        self.y = -0.5*self.g*t**2 + self.vy0*t + self.y0
        
        return (self.x +random.randn()*self.noise[0], self.y +random.randn()*self.noise[1])
        
        

class BallEuler(object):
    def __init__(self, y=100., vel=10., omega=0):
        self.x = 0.
        self.y = y
        omega = radians(omega)
        self.vel = vel*np.cos(omega)
        self.y_vel = vel*np.sin(omega)



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
    
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.

    
        
    
def fx(x,t):
    return fx.vel
    
def fy(y,t):
    return fy.vel - 9.8*t
     
     
class BallRungeKutta(object):
    def __init__(self, x=0, y=100., vel=10., omega = 0.0):   
        self.x = x
        self.y = y
        self.t = 0
        
        omega = math.radians(omega)

        fx.vel = math.cos(omega) * vel
        fy.vel = math.sin(omega) * vel

    def step (self, dt):
        self.x = rk4 (self.x, self.t, dt, fx)
        self.y = rk4 (self.y, self.t, dt, fy)
        self.t += dt 
        print(fx.vel)
        return (self.x, self.y)
        

def ball_scipy(y0, vel, omega, dt):
    
    vel_y = math.sin(math.radians(omega)) * vel
     
    def f(t,y):
        return vel_y-9.8*t
        
    solver = ode(f).set_integrator('dopri5')
    solver.set_initial_value(y0)
 
    ys = [y0]
    while brk.y >= 0:
        t += dt
        brk.step (dt)

        ys.append(solver.integrate(t))
        
        
def RK4(f):
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )( dt * f( t + dt  , y + dy3   ) )
	    )( dt * f( t + dt/2, y + dy2/2 ) )
	    )( dt * f( t + dt/2, y + dy1/2 ) )
	    )( dt * f( t       , y         ) )
 
def theory(t): return (t**2 + 4)**2 /16
 
from math import sqrt
dy = RK4(lambda t, y: t*sqrt(y))
 
t, y, dt = 0., 1., .1
while t <= 10:
    if abs(round(t) - t) < 1e-5:
        print("y(%2.1f)\t= %4.6f \t error: %4.6g" % (t, y, abs(y - theory(t))))
 
    t, y = t + dt, y + dy(t, y, dt)        
        
t = 0.
y=1.

def test(y, t):
    return t*sqrt(y)
    
while t <= 10:
    if abs(round(t) - t) < 1e-5:
        print("y(%2.1f)\t= %4.6f \t error: %4.6g" % (t, y, abs(y - theory(t))))
 
    y = rk4(y, t, dt, test)
    t += dt     
    
if __name__ == "__main__":
    1/0

    dt = 1./30
    y0 = 15.
    vel = 100.
    omega = 30.
    vel_y = math.sin(math.radians(omega)) * vel
    
    def f(t,y):
        return vel_y-9.8*t
        
    be = BallEuler (y=y0, vel=vel,omega=omega)
    #be = BallTrajectory2D (x0=0, y0=y0, velocity=vel, theta_deg = omega)
    ball_rk = BallRungeKutta (y=y0, vel=vel, omega=omega)

    while be.y >= 0:
        be.step (dt)
        ball_rk.step(dt)
            
    print (ball_rk.y - be.y)
        
'''
        p1 = plt.scatter (be.x, be.y, color='red')
        p2 = plt.scatter (ball_rk.x, ball_rk.y, color='blue', marker='v')
    
        plt.legend([p1,p2], ['euler', 'runge kutta'])
'''