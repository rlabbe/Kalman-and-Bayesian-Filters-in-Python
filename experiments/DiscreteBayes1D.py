# -*- coding: utf-8 -*-
"""
Created on Fri May  2 11:48:55 2014

@author: rlabbe
"""
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

''' should this be a class? seems like both sense and update are very
problem specific
'''


def bar_plot(pos, ylim=(0,1), title=None):
    plt.cla()
    ax = plt.gca()
    x = np.arange(len(pos))
    ax.bar(x, pos, color='#30a2da')
    if ylim:
        plt.ylim(ylim)
    plt.xticks(x+0.4, x)
    if title is not None:
        plt.title(title)
        
        
class DiscreteBayes1D(object):

    def __init__(self, world_map, belief=None):
        self.world_map = copy.deepcopy(world_map)
        self.N = len(world_map)

        if belief is None:
            # create belief, make all equally likely
            self.belief = np.empty(self.N)
            self.belief.fill (1./self.N)

        else:
            self.belief = copy.deepcopy(belief)

        # This will be used to temporarily store calculations of the new
        # belief. 'k' just means 'this iteration'.
        self.belief_k = np.empty(self.N)

        assert self.belief.shape == self.world_map.shape


    def _normalize (self):
        s = sum(self.belief)
        self.belief = self.belief/s

    def sense(self, Z, pHit, pMiss):
        for i in range (self.N):
            hit = (self.world_map[i] ==Z)
            self.belief_k[i] = self.belief[i] * (pHit*hit + pMiss*(1-hit))

        # copy results to the belief vector using swap-copy idiom
        self.belief, self.belief_k = self.belief_k, self.belief
        self._normalize()

    def update(self, U, kernel):
        N = self.N
        kN = len(kernel)
        width = int((kN - 1) / 2)

        self.belief_k.fill(0)

        for i in range(N):
            for k in range (kN):
                index = (i + (width-k)-U) % N
                #print(i,k,index)
                self.belief_k[i] += self.belief[index] * kernel[k]

        # copy results to the belief vector using swap-copy idiom
        self.belief, self.belief_k = self.belief_k, self.belief

def add_noise (Z, count):
    n= len(Z)
    for i in range(count):
        j = random.randint(0,n)
        Z[j] = random.randint(0,2)


def animate_three_doors (loops=5):
    world = np.array([1,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0])
    #world = np.array([1,1,1,1,1])
    #world = np.array([1,0,1,0,1,0])


    f = DiscreteBayes1D(world)

    measurements = np.tile(world,loops)
    add_noise(measurements, 4)

    for m in measurements:
        f.sense (m, .8, .2)
        f.update(1, (.05, .9, .05))

        bar_plot(f.belief)
        plt.pause(0.01)


def _test_filter():
    def is_near_equal(a,b):
        try:
            assert sum(abs(a-b)) < 0.001
        except:
            print(a, b)
            assert False

    def test_update_1():
        f = DiscreteBayes1D(np.array([0,0,1,0,0]), np.array([0,0,.8,0,0]))
        f.update (1, [1])
        is_near_equal (f.belief, np.array([0,0,0,.8,0]))

        f.update (1, [1])
        is_near_equal (f.belief, np.array([0,0,0,0,.8]))

        f.update (1, [1])
        is_near_equal (f.belief, np.array([.8,0,0,0,0]))

        f.update (-1, [1])
        is_near_equal (f.belief, np.array([0,0,0,0,.8]))

        f.update (2, [1])
        is_near_equal (f.belief, np.array([0,.8,0,0,0]))

        f.update (5, [1])
        is_near_equal (f.belief, np.array([0,.8,0,0,0]))


    def test_undershoot():
        f = DiscreteBayes1D(np.array([0,0,1,0,0]), np.array([0,0,.8,0,0]))
        f.update (2, [.2, .8,0.])
        is_near_equal (f.belief, np.array([0,0,0,.16,.64]))

    def test_overshoot():
        f = DiscreteBayes1D(np.array([0,0,1,0,0]), np.array([0,0,.8,0,0]))
        f.update (2, [0, .8, .2])
        is_near_equal (f.belief, np.array([.16,0,0,0,.64]))


    test_update_1()
    test_undershoot()

if __name__ == "__main__":

    _test_filter()



    animate_three_doors(loops=1)




