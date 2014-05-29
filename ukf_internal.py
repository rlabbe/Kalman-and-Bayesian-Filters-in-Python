# -*- coding: utf-8 -*-
"""
Created on Tue May 27 21:21:19 2014

@author: rlabbe
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Arrow
import stats
import numpy as np


def arrow(x1,y1,x2,y2):
    return Arrow(x1,y1, x2-x1, y2-y1, lw=2, width=0.1, ec='k', color='k')

def show_2d_transform():
    ax=plt.gca()

    ax.add_artist(Ellipse(xy=(2,5), width=2, height=3,angle=70,linewidth=1,ec='k'))
    ax.add_artist(Ellipse(xy=(7,5), width=2.2, alpha=0.3, height=3.8,angle=150,linewidth=1,ec='k'))

    ax.add_artist(arrow(2, 5, 6, 4.8))
    ax.add_artist(arrow(1.5, 5.5, 7, 3.8))
    ax.add_artist(arrow(2.3, 4.1, 8, 6))

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.axis('equal')
    plt.xlim(0,10); plt.ylim(0,10)
    plt.show()


def show_3_sigma_points():
    xs = np.arange(-4, 4, 0.1)
    var = 1.5
    ys = [stats.gaussian(x, 0, var) for x in xs]
    samples = [0, 1.2, -1.2]
    for x in samples:
        plt.scatter ([x], [stats.gaussian(x, 0, var)], s=80)

    plt.plot(xs, ys)
    plt.show()

def show_sigma_selections():
    ax=plt.gca()
    ax.add_artist(Ellipse(xy=(2,5), alpha=0.5, width=2, height=3,angle=0,linewidth=1,ec='k'))
    ax.add_artist(Ellipse(xy=(5,5), alpha=0.5, width=2, height=3,angle=0,linewidth=1,ec='k'))
    ax.add_artist(Ellipse(xy=(8,5), alpha=0.5, width=2, height=3,angle=0,linewidth=1,ec='k'))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.scatter([1.5,2,2.5],[5,5,5],c='k', s=50)
    plt.scatter([2,2],[4.5, 5.5],c='k', s=50)

    plt.scatter([4.8,5,5.2],[5,5,5],c='k', s=50)
    plt.scatter([5,5],[4.8, 5.2],c='k', s=50)

    plt.scatter([7.2,8,8.8],[5,5,5],c='k', s=50)
    plt.scatter([8,8],[4,6],c='k' ,s=50)

    plt.axis('equal')
    plt.xlim(0,10); plt.ylim(0,10)
    plt.show()







if __name__ == '__main__':
    show_sigma_selections()

