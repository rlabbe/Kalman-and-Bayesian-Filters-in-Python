# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 18:02:07 2013

@author: rlabbe
"""

import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import numpy.random as random

class KalmanFilter:

    def __init__(self, dim):
        self.x = 0 # estimate
        self.P = np.matrix(np.eye(dim)) # uncertainty covariance
        self.Q = 0 # motion uncertainty
        self.u = np.matrix(np.zeros((dim,1))) # motion vector
        self.F = 0 # state transition matrix
        self.H = 0 # Measurment function (maps state to measurements)
        self.R = np.matrix(np.eye(1)) # state uncertainty
        self.I = np.matrix(np.eye(dim))


    def measure(self, Z, R=None):
        """
        Add a new measurement with an optional state uncertainty (R).
        The state uncertainty does not alter the class's state uncertainty,
        it is used only for this call.
        """
        if R is None: R = self.R

        # measurement update
        y = Z - (self.H * self.x)                   # error (residual) between measurement and prediction
        S = (self.H * self.P * self.H.T) + R       # project system uncertainty into measurment space + measurement noise(R)


        K = self.P * self.H.T * linalg.inv(S) # map system uncertainty into kalman gain

        self.x = self.x + (K*y)                # predict new x with residual scaled by the kalman gain
        self.P = (self.I - (K*self.H))*self.P  # and compute the new covariance

    def predict(self):
        # prediction
        self.x = (self.F*self.x) + self.u
        self.P = self.F * self.P * self.F.T + self.Q


if __name__ == "__main__":
    f = KalmanFilter (dim=2)

    f.x = np.matrix([[200.], [0.]])    # initial np.matrix([[z],[0.]],dtype=float)state (location and velocity)
    f.F = np.matrix([[1.,1.],[0.,1.]]) # state transition matrix
    f.H = np.matrix([[1.,0.]])       # Measurement function
    f.R *= 5                       # state uncertainty
    f.P *= 1000.                    # covariance matrix
    f.R = 5
    f.Q = 0




    ps = []
    zs = []
    for t in range (1000):
        z = t + random.randn()*10
        f.measure (z)
        f.predict()
        ps.append (f.x[0,0])
        zs.append(z)
    plt.plot (ps)
   # plt.ylim([0,2])
    plt.plot(zs)
    plt.show()