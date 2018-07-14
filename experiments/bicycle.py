# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:19:21 2015

@author: Roger
"""


from math import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

wheelbase = 100 #inches

vel = 20 *12  # fps to inches per sec
steering_angle = radians(1)
t = 1 # second
orientation = 0. # radians

pos = np.array([0., 0.]

for i in range(100):
#if abs(steering_angle) > 1.e-8:
    turn_radius = tan(steering_angle)
    radius = wheelbase / tan(steering_angle)

    dist = vel*t
    arc_len = dist / (2*pi*radius)

    turn_angle = 2*pi * arc_len


    cx = pos[0] - radius * sin(orientation)
    cy = pos[1] + radius * cos(orientation)

    orientation = (orientation + turn_angle) % (2.0 * pi)
    pos[0] = cx + (sin(orientation) * radius)
    pos[1] = cy - (cos(orientation) * radius)

    plt.scatter(pos[0], pos[1])

plt.axis('equal')




