# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 08:26:08 2015

@author: rlabbe
"""

import numpy as np
import matplotlib.pyplot as plt

from math import sin, cos, factorial
def df(x, p):
    if p == 0:
        return sin(x)

        return x
    if p % 4 == 1:
        return cos(x)

    if p % 4 == 2:
        return -sin(x)

    if p % 4 == 3:
        return -cos(x)

    return sin(x)




def taylor(df, x, a, n):

    f = 0.0

    for i in range(n+1):
        term = df(a, i) * (x - a)**i / factorial(i)
        f += term

    return f


x = 0.1
a = 0.8
n = 1



plt.cla()

xs = np.linspace(-2, 2, 100)
ts = [taylor(df, i, a, n) for i in xs]

plt.plot(xs, np.sin(xs))
plt.plot(xs, ts)


