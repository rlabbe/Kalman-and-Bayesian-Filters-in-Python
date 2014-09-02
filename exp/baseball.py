# -*- coding: utf-8 -*-


""" 
Computes the trajectory of a stitched baseball with air drag.
Takes altitude into account (higher altitude means further travel) and the
stitching on the baseball influencing drag. 

This is based on the book Computational Physics by N. Giordano.

The takeaway point is that the drag coefficient on a stitched baseball is 
LOWER the higher its velocity, which is somewhat counterintuitive.
"""


from __future__ import division
import math
import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np


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



def a_drag (vel, altitude):
    """ returns the drag coefficient of a baseball at a given velocity (m/s)
    and altitude (m).
    """
    
    v_d = 35
    delta = 5
    y_0 = 1.0e4
    
    val = 0.0039 + 0.0058 / (1 + math.exp((vel - v_d)/delta))
    val *= math.exp(-altitude / y_0)
    
    return val

def compute_trajectory_vacuum (v_0_mph, 
                               theta, 
                               dt=0.01, 
                               noise_scale=0.0, 
                               x0=0., y0 = 0.):
    theta = math.radians(theta)
        
    x = x0
    y = y0
    
    v0_y = v_0_mph * math.sin(theta)
    v0_x = v_0_mph * math.cos(theta)    
    
    xs = []
    ys = []
    
    t = dt
    while y >= 0:
        x = v0_x*t + x0
        y = -0.5*9.8*t**2 + v0_y*t + y0
        
        xs.append (x + randn() * noise_scale)        
        ys.append (y + randn() * noise_scale)
        
        t += dt
        
    return (xs, ys)
        
        

def compute_trajectory(v_0_mph, theta, v_wind_mph=0, alt_ft = 0.0, dt = 0.01):
    g = 9.8
    
    ### comput
    theta = math.radians(theta)
    # initial velocity in direction of travel
    v_0 = v_0_mph * 0.447 # mph to m/s
    
    # velocity components in x and y
    v_x = v_0 * math.cos(theta)
    v_y = v_0 * math.sin(theta)
   
    v_wind = v_wind_mph * 0.447 # mph to m/s
    altitude = alt_ft / 3.28   # to m/s
    
    ground_level = altitude
    
    x = [0.0]
    y = [altitude]
    
    i = 0
    while y[i] >= ground_level:
        
        v = math.sqrt((v_x - v_wind) **2+ v_y**2)
        
        x.append(x[i] + v_x * dt)
        y.append(y[i] + v_y * dt)
        
        v_x = v_x - a_drag(v, altitude) * v * (v_x - v_wind) * dt
        v_y = v_y - a_drag(v, altitude) * v * v_y * dt - g * dt
        i += 1
        
    # meters to yards
    np.multiply (x, 1.09361)
    np.multiply (y, 1.09361)
    
    return (x,y)
    

def predict (x0, y0, x1, y1, dt, time):
    g = 10.724*dt*dt
    
    v_x = x1-x0
    v_y = y1-y0    
    v = math.sqrt(v_x**2 + v_y**2)
    
    x = x1
    y = y1
    
    
    while time > 0:
        v_x = v_x - a_drag(v, y) * v * v_x
        v_y = v_y - a_drag(v, y) * v * v_y - g
       
        x = x + v_x
        y = y + v_y

        time -= dt
        
    return (x,y)

        
    
 
if __name__ == "__main__":
    x,y = compute_trajectory(v_0_mph = 110., theta=35., v_wind_mph = 0., alt_ft=5000.)
    
    plt.plot (x, y)
    plt.show()
    

    

    