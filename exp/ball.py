# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 16:07:29 2014

@author: rlabbe
"""

import numpy as np
from KalmanFilter import KalmanFilter
from math import radians, sin, cos, sqrt, exp
import numpy.random as random
import matplotlib.markers as markers
import matplotlib.pyplot as plt


class BallPath(object):
    def __init__(self, x0, y0, omega_deg, velocity, g=9.8, noise=[1.0,1.0]):
        omega = radians(omega_deg)
        self.vx0 = velocity * cos(omega)
        self.vy0 = velocity * sin(omega)

        self.x0 = x0
        self.y0 = y0

        self.g = g
        self.noise = noise

    def pos_at_t(self, t):
        """ returns (x,y) tuple of ball position at time t"""
        x = self.vx0*t + self.x0
        y = -0.5*self.g*t**2 + self.vy0*t + self.y0

        return (x +random.randn()*self.noise[0], y +random.randn()*self.noise[1])



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



def test_baseball_path():
    ball = BaseballPath (0, 1, 35, 50)
    while ball.y > 0:
        ball.update (0.1, 0.)
        plt.scatter (ball.x, ball.y)
        
        
def test_ball_path():
    
    y = 15
    x = 0
    omega = 0.
    noise = [1,1]
    v0 = 100.
    ball = BallPath (x0=x, y0=y, omega_deg=omega, velocity=v0, noise=noise)
    dt = 1
    
    
    f1 = KalmanFilter(dim=6)
    dt = 1/30.   # time step
    
    ay = -.5*dt**2
    
    f1.F = np.mat ([[1, dt,  0,  0,  0,  0],   # x=x0+dx*dt
                    [0,  1, dt,  0,  0,  0],   # dx = dx
                    [0,  0,  0,  0,  0,  0],   # ddx = 0
                    [0,  0,  0,  1, dt, ay],   # y = y0 +dy*dt+1/2*g*dt^2
                    [0,  0,  0,  0,  1, dt],   # dy = dy0 + ddy*dt
                    [0,  0,  0,  0,  0, 1]])  # ddy = -g
    f1.B = 0.
    
    f1.H = np.mat([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0]])
    
    f1.R = np.eye(2) * 5
    f1.Q = np.eye(6) * 0.
    
    omega = radians(omega)
    vx = cos(omega) * v0
    vy = sin(omega) * v0
    
    f1.x = np.mat([x,vx,0,y,vy,-9.8]).T
    
    f1.P = np.eye(6) * 500.
   
    z = np.mat([[0,0]]).T
    count = 0
    markers.MarkerStyle(fillstyle='none')
    
    np.set_printoptions(precision=4)
    while f1.x[3,0] > 0:
        count += 1
        #f1.update (z)
        f1.predict()
        plt.scatter(f1.x[0,0],f1.x[3,0], color='green')


if __name__ == '__main__':
    test_baseball_path()
    #test_ball_path()


