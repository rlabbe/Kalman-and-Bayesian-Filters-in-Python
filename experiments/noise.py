# -*- coding: utf-8 -*-
"""
Created on Fri May  2 10:28:27 2014

@author: rlabbe
"""
import numpy.random
def white_noise (sigma2=1.):
    return sigma2 * numpy.random.randn()


if __name__ == "__main__":
    assert white_noise(.0) == 0.