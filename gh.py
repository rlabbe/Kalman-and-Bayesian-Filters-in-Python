# -*- coding: utf-8 -*-
"""
Created on Tue May 13 13:10:50 2014

@author: RL
"""

# position, velocity (fps)
pos = 20000
vel = 200


t = 10
z = 22060
    
 
def predict_dz (vel,t):
    return vel*t
    

dz = z - pos


print dz - predict_dz(vel,t)


h = 0.1
vel = vel + h * (dz - predict_dz(vel,t)) / t
print 'new vel =', vel
