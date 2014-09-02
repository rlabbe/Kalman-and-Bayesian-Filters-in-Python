# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:35:19 2014

@author: rlabbe
"""
import numpy.random as random
import math

class dog_sensor(object):
    def __init__(self, x0 = 0, motion=1, noise=0.0):
        self.x      = x0
        self.motion = motion
        self.noise  = math.sqrt(noise)

    def sense(self):
        self.x = self.x + self.motion
        self.x += random.randn() * self.noise
        return self.x


def measure_dog ():
    if not hasattr(measure_dog, "x"):
        measure_dog.x = 0
        measure_dog.motion = 1


if __name__ == '__main__':

    dog = dog_sensor(noise = 1)
    for i in range(10):
        print (dog.sense())




