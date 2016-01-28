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

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import JulierSigmaPoints
from math import atan2
import numpy as np

def hx(x):
    """ compute measurements corresponding to state x"""
    dx = x[0] - hx.radar_pos[0]
    dy = x[1] - hx.radar_pos[1]
    return np.array([atan2(dy,dx), (dx**2 + dy**2)**.5])

def fx(x,dt):
    """ predict state x at 'dt' time in the future"""
    return x

def set_radar_pos(pos):
    global hx
    hx.radar_pos = pos

def sensor_fusion_kf():
    global hx, fx
    # create unscented Kalman filter with large initial uncertainty
    points = JulierSigmaPoints(2, kappa=2)
    kf = UKF(2, 2, dt=0.1, hx=hx, fx=fx, points=points)
    kf.x = np.array([100, 100.])
    kf.P *= 40
    return kf

