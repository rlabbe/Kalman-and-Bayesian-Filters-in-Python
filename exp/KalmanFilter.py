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

    def __init__(self, dim_x, dim_z, use_short_form=False):
        """ Create a Kalman filter of dimension 'dim', where dimension is the
        number of state variables.
        
        use_short_form will force the filter to use the short form equation
        for computing covariance: (I-KH)P. This is the equation that results
        from deriving the Kalman filter equations. It is efficient to compute
        but not numerically stable for very long runs. The long form,
        implemented in update(), is very slightly suboptimal, and takes longer
        to compute, but is stable. I have chosen to make the long form the
        default behavior. If you really want the short form, either call
        update_short_form() directly, or set 'use_short_form' to true. This
        causes update() to use the short form. So, if you do it this way
        you will never be able to use the long form again with this object.
        """

        self.x = 0 # state
        self.P = np.matrix(np.eye(dim_x)) # uncertainty covariance
        self.Q = np.matrix(np.eye(dim_x)) # process uncertainty
        self.u = np.matrix(np.zeros((dim_x,1))) # motion vector
        self.B = 0
        self.F = 0 # state transition matrix
        self.H = 0 # Measurement function (maps state to measurements)
        self.R = np.matrix(np.eye(dim_z)) # state uncertainty
        self.I = np.matrix(np.eye(dim_x))
        
        if use_short_form:
            self.update = self.update_short_form


    def update(self, Z):
        """
        Add a new measurement to the kalman filter.
        """

        # measurement update
        y = Z - (self.H * self.x)                   # error (residual) between measurement and prediction
        S = (self.H * self.P * self.H.T) + self.R   # project system uncertainty into measurment space + measurement noise(R)


        K = self.P * self.H.T * linalg.inv(S) # map system uncertainty into kalman gain

        self.x = self.x + (K*y)                # predict new x with residual scaled by the kalman gain

        KH = K*self.H
        I_KH = self.I - KH
        self.P = I_KH*self.P * I_KH.T + K*self.R*K.T


    def update_short_form(self, Z):
        """
        Add a new measurement to the kalman filter.
        
        Uses the 'short form' computation for P, which is mathematically
        correct, but perhaps not that stable when dealing with large data
        sets. But, it is fast to compute. Advice varies; some say to never
        use this. My opinion - if the longer form in update() executes too
        slowly to run in real time, what choice do you have. But that should
        be a rare case, so the long form is the default use
        """

        # measurement update
        y = Z - (self.H * self.x)                   # error (residual) between measurement and prediction
        S = (self.H * self.P * self.H.T) + self.R   # project system uncertainty into measurment space + measurement noise(R)


        K = self.P * self.H.T * linalg.inv(S) # map system uncertainty into kalman gain

        self.x = self.x + (K*y)                # predict new x with residual scaled by the kalman gain

        # and compute the new covariance
        self.P = (self.I - (K*self.H))*self.P  # and compute the new covariance



    def predict(self):
        # prediction
        self.x = (self.F*self.x) + self.B * self.u
        self.P = self.F * self.P * self.F.T + self.Q


if __name__ == "__main__":
    f = KalmanFilter (dim_x=2, dim_z=2)

    f.x = np.matrix([[2.],
                     [0.]])       # initial state (location and velocity)

    f.F = np.matrix([[1.,1.],
                     [0.,1.]])    # state transition matrix

    f.H = np.matrix([[1.,0.]])    # Measurement function
    f.P *= 1000.                  # covariance matrix
    f.R = 5                       # state uncertainty
    f.Q *= 0.0001                 # process uncertainty

    measurements = []
    results = []
    for t in range (100):
        # create measurement = t plus white noise
        z = t + random.randn()*20

        # perform kalman filtering
        f.measure (z)
        f.predict()

        # save data
        results.append (f.x[0,0])
        measurements.append(z)

    # plot data
    p1, = plt.plot(measurements,'r')
    p2, = plt.plot (results,'b')
    p3, = plt.plot ([0,100],[0,100], 'g') # perfect result
    plt.legend([p1,p2, p3], ["noisy measurement", "KF output", "ideal"], 4)


    plt.show()