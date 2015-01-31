# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 18:36:36 2015

@author: rlabbe
"""

import numpy as np
from math import sin, cos, atan2, asin

'''
psi = yaw
phi = roll
theta = pitch
'''


def e2q(vector):
    
    roll    = vector[0]
    pitch   = vector[1]
    heading = vector[2]
    sinhdg = sin(heading/2)
    coshdg = cos(heading/2)
    
    sinroll = sin(roll/2)
    cosroll = cos(roll/2)
    
    sinpitch = sin(pitch/2)
    cospitch = cos(pitch/2)
    
    q0 = cosroll*cospitch*coshdg + sinroll*sinpitch*sinhdg
    q1 = sinroll*cospitch*coshdg - cosroll*sinpitch*sinhdg
    q2 = cosroll*sinpitch*coshdg + sinroll*cospitch*sinhdg
    q3 = cosroll*cospitch*sinhdg - sinroll*sinpitch*coshdg
    
    return np.array([q0, q1, q2, q3])
    
    
def q2e(q):
    roll = atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
    pitch = asin(2*(q[0]*q[2] - q[3]*q[1]))
    hdg = atan2(2*(q[0]*q[3] + q[1]*q[2]), 1-2*(q[2]**2 + q[3]**2))
    
    return np.array([roll, pitch, hdg])
   
   
def e2d(e):
    return np.degrees(e)
def e2r(e):
    return np.radians(e)
    
def add(q1,q2):
    return np.multiply(q1,q2)



def add2(q1, q2):
    Q1 = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    Q2 = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    Q3 = q1[0]*q2[2] + q1[2]*q2[0] + q1[3]*q2[1] - q1[1]*q2[3]
    Q4 = q1[0]*q2[3] + q1[3]*q2[0] + q1[1]*q2[2] - q1[2]*q2[1]
    
    return np.array([Q1, Q2, Q3, Q4])


e = e2r([10, 0, 0])
q = e2q(e)
print(q)
print(e2d(q2e(q)))
q2 = add2(q,q)
print(q2)
e2 = q2e(q2)
print(e2d(e2))

    