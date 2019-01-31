# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:07:26 2014

@author: RL
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy.random as random






def g_h_filter (data, x, dx, g, h, dt=1.):
    results = []
    for z in data:
        x_est = x + (dx*dt)
        residual = z - x_est

        dx = dx    + h * (residual / float(dt))
        x  = x_est + g * residual
        print('gx',dx,)

        results.append(x)


    return results


'''
computation of x
x_est = weight + gain
residual = z - weight - gain
x  = weight + gain + g (z - weight - gain)

w + gain + gz -wg -ggain
w -wg + gain - ggain + gz

w(1-g) + gain(1-g) +gz

(w+g)(1-g) +gz

'''
'''
gain computation

gain = gain + h/t* (z - weight - gain)
= gain + hz/t -hweight/t - hgain/t

= gain(1-h/t) + h/t(z-weight)

'''
'''
gain+ h*(z-w -gain*t)/t

gain + hz/t -hw/t -hgain

gain*(1-h) + h/t(z-w)


'''
def weight2():
    w = 0
    gain = 200
    t = 10.
    weight_scale = 1./6
    gain_scale = 1./10

    weights=[2060]
    for i in range (len(weights)):
        z = weights[i]
        w_pre = w + gain*t

        new_w = w_pre * (1-weight_scale) +  z * (weight_scale)

        print('new_w',new_w)

        gain = gain *(1-gain_scale) + (z - w) * gain_scale/t

        print (z)
        print(w)

        #gain = new_gain * (gain_scale) + gain * (1-gain_scale)
        w = new_w
        print ('w',w,)
        print ('gain=',gain)


def weight3():
    w = 160.
    gain = 1.
    t = 1.
    weight_scale = 6/10.
    gain_scale = 2./3

    weights=[158]
    for i in range (len(weights)):
        z = weights[i]
        w_pre = w + gain*t

        new_w = w_pre * (1-weight_scale) +  z * (weight_scale)

        print('new_w',new_w)

        gain = gain *(1-gain_scale) + (z - w) * gain_scale/t

        print (z)
        print(w)

        #gain = new_gain * (gain_scale) + gain * (1-gain_scale)
        w = new_w
        print ('w',w,)
        print ('gain=',gain)
weight3()
'''
#zs = [i + random.randn()*50 for i in range(200)]
zs = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0]

#zs = [2060]
data= g_h_filter(zs, 160, 1, .6, 0, 1.)

'''

data = g_h_filter([2060], x=0, dx=200, g=1./6, h = 1./10, dt=10)
print data


'''
print data
print data2
plt.plot(data)
plt.plot(zs, 'g')
plt.show()
'''
