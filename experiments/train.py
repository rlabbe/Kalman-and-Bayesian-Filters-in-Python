# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:02:39 2015

@author: Roger
"""


import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt


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



def normalize(belief):
    s = sum(belief)
    belief /= s

def update(map_, belief, z, p_hit, p_miss):
    for i, val in enumerate(map_):
        if val == z:
            belief[i] *= p_hit
        else:
            belief[i] *= p_miss

    belief = normalize(belief)


def predict(prob_dist, offset, kernel):
    N = len(prob_dist)
    kN = len(kernel)
    width = int((kN - 1) / 2)

    result = np.zeros(N)
    for i in range(N):
        for k in range (kN):
            index = (i + (width-k) - offset) % N
            result[i] += prob_dist[index] * kernel[k]
    prob_dist[:] = result[:] # update belief


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

    measurements = np.tile(world, loops)
    add_noise(measurements, 50)

    for m in measurements:
        f.sense (m, .8, .2)
        f.update(1, (.05, .9, .05))

        bar_plot(f.belief)
        plt.pause(0.01)


def animate_book(loops=5):
    world = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    #world = np.array([1,1,1,1,1])
    #world = np.array([1,0,1,0,1,0])

    N = len(world)
    belief = np.array([1./N]*N)

    measurements = np.tile(world, loops)
    add_noise(measurements, 5)

    for m in measurements:
        update(world, belief, m, .8, .2)
        predict(belief, 1, (.05, .9, .05))

        bar_plot(belief)
        plt.pause(0.01)
    print(f.belief)


import random

class Train(object):

    def __init__(self, track, kernel=[1.], sense_error=.1, no_sense_error=.05):
        self.track = track
        self.pos = 0
        self.kernel = kernel
        self.sense_error = sense_error
        self.no_sense_error = no_sense_error


    def move(self, distance=1):
        """ move in the specified direction with some small chance of error"""

        self.pos += distance

        # insert random movement error according to kernel
        r = random.random()
        s = 0
        offset = -(len(self.kernel) - 1) / 2
        for k in self.kernel:
            s += k
            if r <= s:
                break
            offset += 1

        self.pos = (self.pos + offset) % len(self.track)
        return self.pos

    def sense(self):
        pos = self.pos

         # insert random sensor error
        r = random.random()
        if r < self.sense_error:
            if random.random() > 0.5:
                pos += 1
            else:
                pos -= 1
                print('sense error')
        return pos


def animate_train(loops=5):
    world = np.array([1,2,3,4,5,6,7,8,9,10])

    N = len(world)
    belief = np.zeros(N)
    belief[0] = 1.0

    robot = Train(world, [.1, .8, .1], .1, .1)

    for i in range(N*loops):
        robot.move(1)
        m = robot.sense()
        update(world, belief, m, .9, .1)
        predict(belief, 1, (.05, .9, .05))

        bar_plot(belief)
        plt.pause(0.1)
    print(belief)

animate_train(3)

world = np.array([1,2,3,4,5,6,7,8,9,10])
#world = np.array([1,1,1,1,1])
#world = np.array([1,0,1,0,1,0])


