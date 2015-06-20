# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:16:54 2015

@author: rlabbe
"""


from filterpy.kalman import KalmanFilter



kf = KalmanFilter(1, 1)
kf.x[0,0] = 1.
kf.P = np.ones((1,1))
kf.H = np.array([[2.]])
kf.F = np.ones((1,1))
kf.R = 1
kf.Q = 0



kf.predict()
kf.update(2)

