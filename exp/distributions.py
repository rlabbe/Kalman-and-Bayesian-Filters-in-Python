# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 20:17:14 2015

@author: rlabbe
"""

from scipy.stats import t, norm
import matplotlib.pyplot as plt
import numpy as np
import math
import random


df =4
mu = 10
std = 2

x = np.linspace(-5, 20, 100)
               
plt.plot(x, t.pdf(x, df=df, loc=mu, scale=std), 'r-', lw=5, label='t pdf')


x2 = np.linspace(mu-10, mu+10, 100)
plt.plot(x, norm.pdf(x, mu, std), 'b-', lw=5,  label='gaussian pdf')
plt.legend()
plt.figure()

def student_t(df, mu, std): # nu equals number of degrees of freedom
    x = random.gauss(0, std)
    y = 2.0*random.gammavariate(0.5*df, 2.0)
    return x / (math.sqrt(y/df)) + mu
    
    
N = 100000
ys = [student_t(2.7, 100, 2) for i in range(N)]
plt.hist(ys, 10000, histtype='step')

ys = [random.gauss(100,2) for i in range(N)]
plt.hist(ys, 10000, histtype='step',color='r')

plt.show()