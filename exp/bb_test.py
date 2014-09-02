# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:33:38 2014

@author: rlabbe
"""

from __future__ import print_function,division
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import baseball
from numpy.random import randn

def ball_filter6(dt,R=1., Q = 0.1):
    f1 = KalmanFilter(dim=6)
    g = 10

    f1.F = np.mat ([[1., dt, dt**2,  0,       0,  0],
                    [0,  1., dt,     0,       0,  0],
                    [0,  0,  1.,     0,       0,  0],
                    [0,  0,  0.,    1., dt, -0.5*dt*dt*g],
                    [0,  0,  0,      0, 1.,      -g*dt],
                    [0,  0,  0,      0, 0.,      1.]])

    f1.H = np.mat([[1,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,1,0,0],
                   [0,0,0,0,0,0],
                   [0,0,0,0,0,0]])


    f1.R = np.mat(np.eye(6)) * R

    f1.Q = np.zeros((6,6))
    f1.Q[2,2] = Q
    f1.Q[5,5] = Q
    f1.x = np.mat([0, 0 , 0, 0, 0, 0]).T
    f1.P = np.eye(6) * 50.
    f1.B = 0.
    f1.u = 0

    return f1


def ball_filter4(dt,R=1., Q = 0.1):
    f1 = KalmanFilter(dim=4)
    g = 10

    f1.F = np.mat ([[1., dt,  0, 0,],
                    [0,  1.,  0, 0],
                    [0,  0,  1., dt],
                    [0,  0,  0.,  1.]])

    f1.H = np.mat([[1,0,0,0],
                   [0,0,0,0],
                   [0,0,1,0],
                   [0,0,0,0]])



    f1.B = np.mat([[0,0,0,0],
                   [0,0,0,0],
                   [0,0,1.,0],
                   [0,0,0,1.]])

    f1.u = np.mat([[0],
                   [0],
                   [-0.5*g*dt**2],
                   [-g*dt]])

    f1.R = np.mat(np.eye(4)) * R

    f1.Q = np.zeros((4,4))
    f1.Q[1,1] = Q
    f1.Q[3,3] = Q
    f1.x = np.mat([0, 0 , 0, 0]).T
    f1.P = np.eye(4) * 50.
    return f1


def plot_ball_filter6 (f1, zs, skip_start=-1, skip_end=-1):
    xs, ys = [],[]
    pxs, pys = [],[]

    for i,z in enumerate(zs):
        m = np.mat([z[0], 0, 0, z[1], 0, 0]).T

        f1.predict ()

        if i == skip_start:
            x2 = xs[-2]
            x1 = xs[-1]
            y2 = ys[-2]
            y1 = ys[-1]

        if i >= skip_start and i <= skip_end:
            #x,y = baseball.predict (x2, y2, x1, y1, 1/30, 1/30* (i-skip_start+1))
            x,y = baseball.predict (xs[-2], ys[-2], xs[-1], ys[-1], 1/30, 1/30)

            m[0] = x
            m[3] = y
        #print ('skip', i, f1.x[2],f1.x[5])

        f1.update(m)


        '''
        if i >= skip_start and i <= skip_end:
            #f1.x[2] = -0.1
            #f1.x[5] = -17.
            pass
        else:
            f1.update (m)

        '''

        xs.append (f1.x[0,0])
        ys.append (f1.x[3,0])
        pxs.append (z[0])
        pys.append(z[1])

        if i > 0 and z[1] < 0:
            break;

    p1, = plt.plot (xs, ys, 'r--')
    p2, = plt.plot (pxs, pys)
    plt.axis('equal')
    plt.legend([p1,p2], ['filter', 'measurement'], 2)
    plt.xlim([0,xs[-1]])
    plt.show()



def plot_ball_filter4 (f1, zs, skip_start=-1, skip_end=-1):
    xs, ys = [],[]
    pxs, pys = [],[]

    for i,z in enumerate(zs):
        m = np.mat([z[0], 0, z[1], 0]).T

        f1.predict ()

        if i == skip_start:
            x2 = xs[-2]
            x1 = xs[-1]
            y2 = ys[-2]
            y1 = ys[-1]

        if i >= skip_start and i <= skip_end:
            #x,y = baseball.predict (x2, y2, x1, y1, 1/30, 1/30* (i-skip_start+1))
            x,y = baseball.predict (xs[-2], ys[-2], xs[-1], ys[-1], 1/30, 1/30)

            m[0] = x
            m[2] = y

        f1.update (m)


        '''
        if i >= skip_start and i <= skip_end:
            #f1.x[2] = -0.1
            #f1.x[5] = -17.
            pass
        else:
            f1.update (m)

        '''

        xs.append (f1.x[0,0])
        ys.append (f1.x[2,0])
        pxs.append (z[0])
        pys.append(z[1])

        if i > 0 and z[1] < 0:
            break;

    p1, = plt.plot (xs, ys, 'r--')
    p2, = plt.plot (pxs, pys)
    plt.axis('equal')
    plt.legend([p1,p2], ['filter', 'measurement'], 2)
    plt.xlim([0,xs[-1]])
    plt.show()


start_skip = 20
end_skip = 60

def run_6():
    dt = 1/30
    noise = 0.0
    f1 = ball_filter6(dt, R=.16, Q=0.1)
    plt.cla()

    x,y = baseball.compute_trajectory(v_0_mph = 100., theta=50., dt=dt)


    znoise = [(i+randn()*noise,j+randn()*noise) for (i,j) in zip(x,y)]

    plot_ball_filter6 (f1, znoise, start_skip, end_skip)


def run_4():
    dt = 1/30
    noise = 0.0
    f1 = ball_filter4(dt, R=.16, Q=0.1)
    plt.cla()

    x,y = baseball.compute_trajectory(v_0_mph = 100., theta=50., dt=dt)


    znoise = [(i+randn()*noise,j+randn()*noise) for (i,j) in zip(x,y)]

    plot_ball_filter4 (f1, znoise, start_skip, end_skip)


if __name__ == "__main__":
    run_4()