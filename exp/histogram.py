# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:43:38 2014

@author: rlabbe
"""

p = [.2, .2, .2, .2, .2]
world = ['green', 'red', 'red', 'green', 'green']
measurements = ['red', 'green']

pHit = 0.6
pMiss = 0.2

pOvershoot = 0.1
pUndershoot = 0.1
pExact = 0.8

def normalize (p):
    s = sum(p)
    for i in range (len(p)):
        p[i] = p[i] / s

def sense(p, Z):
    q= []
    for i in range (len(p)):
        hit = (world[i] ==Z)
        q.append(p[i] * (pHit*hit + pMiss*(1-hit)))
    normalize(q)
    return q


def move(p, U):
    q = []
    for i in range(len(p)):
        pexact = p[(i-U) % len(p)] * pExact
        pover  =  p[(i-U-1) % len(p)] * pOvershoot
        punder =  p[(i-U+1) % len(p)] * pUndershoot
        q.append (pexact + pover + punder)

    normalize(q)
    return q

if __name__ == "__main__":

    p = sense(p, 'red')
    print p
    pause()
    for z in measurements:
        p = sense (p, z)
        p = move (p, 1)
        print p





