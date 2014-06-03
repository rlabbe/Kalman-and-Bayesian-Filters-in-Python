# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/rlabbe/.spyder2/.temp.py
"""


""" 

Computes the trajectory of a stitched baseball with air drag.
Takes altitude into account (higher altitude means further travel) and the
stitching on the baseball influencing drag. 

This is based on the book Computational Physics by N. Giordano.

The takeaway point is that the drag coefficient on a stitched baseball is 
LOWER the higher its velocity, which is somewhat counterintuitive.
"""



import math
import matplotlib.pyplot as plt

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



def compute_trajectory(v_0_mph, theta, v_wind_mph=0, alt_ft = 0.0, dt = 0.01):
    ### comput
    theta = math.radians(theta)
    
    v_0 = v_0_mph * 0.447 # mph to m/s
    v_x = v_0 * math.cos(theta)
    v_y = v_0 * math.sin(theta)
   
    v_wind = v_wind_mph * 0.447 # mph to m/s
    altitude = alt_ft / 3.28   # to m/s
    
    ground_level = altitude
    
    x = [0.0]
    y = [altitude]
    
    i = 0
    while y[i] >= ground_level:
        g = 9.8
        
        v = math.sqrt((v_x - v_wind) **2+ v_y**2)
        
        x.append(x[i] + v_x * dt)
        y.append(y[i] + v_y * dt)
        
        v_x = v_x - a_drag(v, altitude) * v * (v_x - v_wind) * dt
        v_y = v_y - a_drag(v, altitude) * v * v_y * dt - g * dt
        i += 1
        
    return (x,y)
 
if __name__ == "__main__":
    x,y = compute_trajectory(v_0_mph = 110., theta=35., v_wind_mph = 0., alt_ft=5000.)
    
    plt.plot (x, y)
    plt.show()
    

    

    