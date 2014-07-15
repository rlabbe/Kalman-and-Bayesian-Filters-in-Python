# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from KalmanFilter import *
from math import cos, sin, sqrt, atan2


def H_of (pos, pos_A, pos_B):
    """ Given the position of our object at 'pos' in 2D, and two transmitters
    A and B at positions 'pos_A' and 'pos_B', return the partial derivative
    of H
    """

    theta_a = atan2(pos_a[1]-pos[1], pos_a[0] - pos[0])
    theta_b = atan2(pos_b[1]-pos[1], pos_b[0] - pos[0])

    if False:
        return np.mat([[0, -cos(theta_a), 0, -sin(theta_a)],
                       [0, -cos(theta_b), 0, -sin(theta_b)]])
    else:                  
        return np.mat([[-cos(theta_a), 0, -sin(theta_a), 0],
                       [-cos(theta_b), 0, -sin(theta_b), 0]])

class DMESensor(object):
    def __init__(self, pos_a, pos_b, noise_factor=1.0):
        self.A = pos_a
        self.B = pos_b
        self.noise_factor = noise_factor

    def range_of (self, pos):
        """ returns tuple containing noisy range data to A and B
        given a position 'pos'
        """

        ra = sqrt((self.A[0] - pos[0])**2 + (self.A[1] - pos[1])**2)
        rb = sqrt((self.B[0] - pos[0])**2 + (self.B[1] - pos[1])**2)

        return (ra + random.randn()*self.noise_factor,
                rb + random.randn()*self.noise_factor)


pos_a = (100,-20)
pos_b = (-100, -20)

f1 = KalmanFilter(dim_x=4, dim_z=2)

f1.F = np.mat ([[0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0]])

f1.B = 0.

f1.R *= 1.
f1.Q *= .1

f1.x = np.mat([1,0,1,0]).T
f1.P = np.eye(4) * 5.

# initialize storage and other variables for the run
count = 30
xs, ys = [],[]
pxs, pys = [],[]

# create the simulated sensor
d = DMESensor (pos_a, pos_b, noise_factor=1.)

# pos will contain our nominal position since the filter does not
# maintain position.
pos = [0,0]

for i in range(count):
    # move (1,1) each step, so just use i
    pos = [i,i]
    
    # compute the difference in range between the nominal track and measured 
    # ranges
    ra,rb = d.range_of(pos)
    rx,ry = d.range_of((i+f1.x[0,0], i+f1.x[2,0]))
    
    print ra, rb
    print rx,ry
    z = np.mat([[ra-rx],[rb-ry]])
    print z.T

    # compute linearized H for this time step
    f1.H = H_of (pos, pos_a, pos_b)

    # store stuff so we can plot it later
    xs.append (f1.x[0,0]+i)
    ys.append (f1.x[2,0]+i)
    pxs.append (pos[0])
    pys.append(pos[1])
    
    # perform the Kalman filter steps
    f1.predict ()
    f1.update(z)


p1, = plt.plot (xs, ys, 'r--')
p2, = plt.plot (pxs, pys)
plt.legend([p1,p2], ['filter', 'ideal'], 2)
plt.show()  
    
