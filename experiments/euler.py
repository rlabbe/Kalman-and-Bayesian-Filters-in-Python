# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:40:47 2015

@author: Roger
"""




def dx(t, y):
    return y
    
    
def euler(t0, tmax, y0, dx, step=1.):
    t = t0
    y = y0
    while t < tmax:
        f = dx(t,y)
        y = y + step*dx(t, y)
        t +=step
        
    return y
    
    
print(euler(0, 4, 1, dx, step=0.25))