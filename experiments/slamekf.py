# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 08:25:01 2016

@author: roger
"""

import numpy as np


class WorldMap(object):
    
    def __init__(self, N=100):
        
        self.N = N
        pass
        
        
        
        
    def measurements(self, x, theta):
        """ return array of measurements (range, angle) if robot is in position
        x"""
    
        N = 10
        a = np.linspace(-np.pi, np.pi, self.N)
        return a
        
        

def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    
    source:
    http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
    """
    # Setup initial conditions
    x1, y1 = int(round(start[0])), int(round(start[1]))
    x2, y2 = int(round(end[0])), int(round(end[1]))
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points 
 

world = np.zeros((1000,1000), dtype=bool)


def add_line(p0, p1):
    pts = get_line(p0, p1)
    for p in pts:
        try:
            world[p[0], p[1]] = True
        except:
            pass # ignore out of range
        
        
add_line((0,0), (1000, 0))

def measure(x, theta):

    dx,dy = world.shape
    h = np.sqrt(2*(dx*dx + dy+dy))
    p1 = [h*np.cos(theta), h*np.sin(theta)]

    
    hits = get_line(x, p1)
    
    try:
        for pt in hits:
            if world[pt[0], pt[1]]:
                return pt
    except:
        return -1
    return -2
    
    
    

measure([100,100], -np.pi/2)
