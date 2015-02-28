# -*- coding: utf-8 -*-


import numpy as np
from numpy.linalg import norm, inv
from numpy.random import randn
from numpy import dot


numpy.random.seed(1234)
user_pos = np.array([1000, 100])  # d5, D6

pred_user_pos = np.array([100, 0]) #d7, d8


t_pos = np.asarray([[0, 1000],
                    [0, -1000],
                    [500, 500]], dtype=float)


def transmitter_range(pos, transmitter_pos):
    """ Compute distance between position 'pos' and the list of positions
    in transmitter_pos"""

    N = len(transmitter_pos)
    rng = np.zeros(N)

    diff = np.asarray(pos) - transmitter_pos

    for i in range(N):
        rng[i] = norm(diff[i])

    return norm(diff, axis=1)




# compute measurement of where you are with respect to seach sensor


rz= transmitter_range(user_pos, t_pos) # $B21,22

# add some noise
for i in range(len(rz)):
    rz[i] += randn()


# now iterate on the predicted position
pos = pred_user_pos


def hx_range(pos, t_pos, r_est):
    N = len(t_pos)
    H = np.zeros((N, 2))
    for j in range(N):
        H[j,0] = -(t_pos[j,0] - pos[0]) / r_est[j]
        H[j,1] = -(t_pos[j,1] - pos[1]) / r_est[j]
    return H


def lop_ils(zs, t_pos, pos_est, hx, eps=1.e-6):
    """ iteratively estimates the solution to a set of measurement, given
    known transmitter locations"""
    pos = np.array(pos_est)

    converged = False
    for i in range(20):
        r_est = transmitter_range(pos, t_pos) #B32-B33
        print('iteration:', i)
        #print ('ra1, ra2', ra1, ra2)
        print()

        H=hx(pos, t_pos, r_est)
        
        Hinv = inv(dot(H.T, H)).dot(H.T)

        #update position estimate
        y = zs - r_est
        print('residual', y)

        Hy = np.dot(Hinv, y)
        print('Hy', Hy)

        pos = pos + Hy
        print('pos', pos)

        print()
        print()

        if max(abs(Hy)) < eps:
            converged = True
            break

    return pos, converged



print(lop_ils(rz, t_pos, (900,90), hx=hx_range))



#####################
"""
# compute measurement (simulation)
rza1, rza2 = transmitter_range(user_pos) # $B21,22

rza1 += randn()
rza2 += randn()

# now iterate on the predicted position
pos = pred_user_pos


for i in range(10):
    ra1, ra2 = transmitter_range(pos) #B32-B33
    print('iteration:', i)
    print ('ra1, ra2', ra1, ra2)
    print()

    H = np.array([[-(t1_pos[0] - pos[0]) / ra1, -(t1_pos[1] - pos[1]) / ra1],
                  [-(t2_pos[0] - pos[0]) / ra2, -(t2_pos[1] - pos[1]) / ra2]])
    Hinv = inv(H)

    #update position estimate
    residual_t1 = rza1 - ra1
    residual_t2 = rza2 - ra2
    y = np.array([[residual_t1], [residual_t2]])
    print('residual', y.T)


    Hy = np.dot(Hinv, y)

    pos = pos + Hy[:,0]
    print('pos', pos)

    print()
    print()

    if (max(abs(y)) < 1.e-6):
        break
"""
