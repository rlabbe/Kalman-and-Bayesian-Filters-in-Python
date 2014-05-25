# -*- coding: utf-8 -*-
"""
Created on Sat May 24 14:42:55 2014

@author: rlabbe
"""

from KalmanFilter import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

f = KalmanFilter (dim=4)

dt = 1
f.F = np.mat ([[1, dt, 0,  0],
               [0,  1, 0,  0],
               [0,  0, 1, dt],
               [0,  0, 0,  1]])

f.H = np.mat ([[1, 0, 0, 0],
               [0, 0, 1, 0]])



f.Q *= 4.
f.R = np.mat([[50,0],
              [0, 50]])

f.x = np.mat([0,0,0,0]).T
f.P *= 100.


xs = []
ys = []
count = 200
for i in range(count):
    z = np.mat([[i+random.randn()*1],[i+random.randn()*1]])
    f.predict ()
    f.update (z)
    xs.append (f.x[0,0])
    ys.append (f.x[2,0])


plt.plot (xs, ys)
plt.show()

