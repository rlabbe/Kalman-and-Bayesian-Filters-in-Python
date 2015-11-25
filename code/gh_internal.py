# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.


Code supporting the book

Kalman and Bayesian Filters in Python
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


This is licensed under an MIT license. See the LICENSE.txt file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import book_format
import book_plots
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon, Arrow, FancyArrow
import pylab as plt

def plot_errorbar1():
    with book_format.figsize(y=1.5):
        book_plots.plot_errorbars([(160, 8, 'A'), (170, 8, 'B')],
                                   xlims=(145, 185), ylims=(-1, 2))

def plot_errorbar2():
    with book_format.figsize(y=1.5):
        book_plots.plot_errorbars([(160, 3, 'A'), (170, 9, 'B')],
                                   xlims=(145, 185), ylims=(-1, 2))

def plot_errorbar3():
    with book_format.figsize(y=1.5):
        book_plots.plot_errorbars([(160, 1, 'A'), (170, 9, 'B')], 
                                  xlims=(145, 185), ylims=(-1, 2))



def plot_gh_results(weights, estimates, predictions):
    n = len(weights)

    xs = list(range(n+1))
    book_plots.plot_filter(xs, estimates, marker='o')
    book_plots.plot_measurements(xs[1:], weights, color='k', label='Scale', lines=False)
    book_plots.plot_track([0, n], [160, 160+n], c='k', label='Actual Weight')
    book_plots.plot_track(xs[1:], predictions, c='r', label='Predictions', marker='v')
    book_plots.show_legend()
    book_plots.set_labels(x='day', y='weight (lbs)')
    plt.xlim([0, n])
    plt.show()


def print_results(estimates, prediction, weight):
    print('previous: {:.2f}, prediction: {:.2f} estimate {:.2f}'.format(
          estimates[-2], prediction, weight))

def create_predict_update_chart(box_bg = '#CCCCCC',
                arrow1 = '#88CCFF',
                arrow2 = '#88FF88'):
    plt.figure(figsize=(4,4), facecolor='w')
    ax = plt.axes((0, 0, 1, 1),
                  xticks=[], yticks=[], frameon=False)
    #ax.set_xlim(0, 10)
    #ax.set_ylim(0, 10)


    pc = Circle((4,5), 0.5, fc=box_bg)
    uc = Circle((6,5), 0.5, fc=box_bg)
    ax.add_patch (pc)
    ax.add_patch (uc)


    plt.text(4,5, "Predict\nStep",ha='center', va='center', fontsize=14)
    plt.text(6,5, "Update\nStep",ha='center', va='center', fontsize=14)

    #btm
    ax.annotate('',
                xy=(4.1, 4.5),  xycoords='data',
                xytext=(6, 4.5), textcoords='data',
                size=20,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none",
                                patchB=pc,
                                patchA=uc,
                                connectionstyle="arc3,rad=-0.5"))
    #top
    ax.annotate('',
                xy=(6, 5.5),  xycoords='data',
                xytext=(4.1, 5.5), textcoords='data',
                size=20,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none",
                                patchB=uc,
                                patchA=pc,
                                connectionstyle="arc3,rad=-0.5"))


    ax.annotate('Measurement ($\mathbf{z_k}$)',
                xy=(6.3, 5.4),  xycoords='data',
                xytext=(6,6), textcoords='data',
                size=18,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    ax.annotate('',
                xy=(4.0, 3.5),  xycoords='data',
                xytext=(4.0,4.5), textcoords='data',
                size=18,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    ax.annotate('Initial\nConditions ($\mathbf{x_0}$)',
                xy=(4.0, 5.5),  xycoords='data',
                xytext=(2.5,6.5), textcoords='data',
                size=18,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    plt.text (4,3.4,'State Estimate ($\mathbf{\hat{x}_k}$)',
              ha='center', va='center', fontsize=18)
    plt.axis('equal')
    #plt.axis([0,8,0,8])
    plt.show()


def plot_estimate_chart_1():
    ax = plt.axes()
    ax.annotate('', xy=[1,159], xytext=[0,158],
                arrowprops=dict(arrowstyle='->', ec='r',shrinkA=6, lw=3,shrinkB=5))
    plt.scatter ([0], [158], c='b')
    plt.scatter ([1], [159], c='r')
    plt.xlabel('day')
    plt.ylabel('weight (lbs)')
    ax.xaxis.grid(True, which="major", linestyle='dotted')
    ax.yaxis.grid(True, which="major", linestyle='dotted')
    plt.show()


def plot_estimate_chart_2():
    ax = plt.axes()
    ax.annotate('', xy=[1,159], xytext=[0,158],
                arrowprops=dict(arrowstyle='->',
                                ec='r', lw=3, shrinkA=6, shrinkB=5))
    plt.scatter ([0], [158.0], c='k',s=128)
    plt.scatter ([1], [164.2], c='b',s=128)
    plt.scatter ([1], [159], c='r', s=128)
    plt.text (1.0, 158.8, "prediction ($x_t)$", ha='center',va='top',fontsize=18,color='red')
    plt.text (1.0, 164.4, "measurement ($z$)",ha='center',va='bottom',fontsize=18,color='blue')
    plt.text (0, 157.8, "estimate ($\hat{x}_{t-1}$)", ha='center', va='top',fontsize=18)
    plt.xlabel('day')
    plt.ylabel('weight (lbs)')
    ax.xaxis.grid(True, which="major", linestyle='dotted')
    ax.yaxis.grid(True, which="major", linestyle='dotted')
    plt.show()

def plot_estimate_chart_3():
    ax = plt.axes()
    ax.annotate('', xy=[1,159], xytext=[0,158],
                arrowprops=dict(arrowstyle='->',
                                ec='r', lw=3, shrinkA=6, shrinkB=5))

    ax.annotate('', xy=[1,159], xytext=[1,164.2],
                arrowprops=dict(arrowstyle='-',
                                ec='k', lw=3, shrinkA=8, shrinkB=8))

    est_y = ((164.2-158)*.8 + 158)
    plt.scatter ([0,1], [158.0,est_y], c='k',s=128)
    plt.scatter ([1], [164.2], c='b',s=128)
    plt.scatter ([1], [159], c='r', s=128)
    plt.text (1.0, 158.8, "prediction ($x_t)$", ha='center',va='top',fontsize=18,color='red')
    plt.text (1.0, 164.4, "measurement ($z$)",ha='center',va='bottom',fontsize=18,color='blue')
    plt.text (0, 157.8, "estimate ($\hat{x}_{t-1}$)", ha='center', va='top',fontsize=18)
    plt.text (0.95, est_y, "new estimate ($\hat{x}_{t}$)", ha='right', va='center',fontsize=18)
    plt.xlabel('day')
    plt.ylabel('weight (lbs)')
    ax.xaxis.grid(True, which="major", linestyle='dotted')
    ax.yaxis.grid(True, which="major", linestyle='dotted')
    plt.show()

def plot_hypothesis():
    plt.errorbar([1, 2, 3], [170, 161, 169],
                 xerr=0, yerr=10, fmt='bo', capthick=2, capsize=10)

    plt.plot([1, 3], [180, 160], color='g', ls='--')
    plt.plot([1, 3], [170, 170], color='g', ls='--')
    plt.plot([1, 3], [160, 175], color='g', ls='--')
    plt.plot([1, 2, 3], [180, 152, 179], color='g', ls='--')
    plt.xlim(0,4); plt.ylim(150, 185)
    plt.xlabel('day')
    plt.ylabel('lbs')
    plt.tight_layout()
    plt.show()

def plot_hypothesis2():
    plt.errorbar(range(1, 11), [169, 170, 169,171, 170, 171, 169, 170, 169, 170],
                 xerr=0, yerr=6, fmt='bo', capthick=2, capsize=10)
    plt.plot([1, 10], [169, 170.5], color='g', ls='--')
    plt.xlim(0, 11); plt.ylim(150, 185)
    plt.xlabel('day')
    plt.ylabel('lbs')
    plt.show()


def plot_hypothesis3():
    weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]
    plt.errorbar(range(1, 13), weights,
                 xerr=0, yerr=6, fmt='o', capthick=2, capsize=10)

    plt.xlim(0, 13); plt.ylim(145, 185)
    plt.xlabel('day')
    plt.ylabel('weight (lbs)')
    plt.show()


def plot_hypothesis4():
    weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

    ave = np.sum(weights) / len(weights)
    plt.errorbar(range(1,13), weights, label='weights',
                 yerr=6, fmt='o', capthick=2, capsize=10)
    plt.plot([1, 12], [ave,ave], c='r', label='hypothesis')
    plt.xlim(0, 13); plt.ylim(145, 185)
    plt.xlabel('day')
    plt.ylabel('weight (lbs)')
    book_plots.show_legend()
    plt.show()


def plot_hypothesis5():
    weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

    xs = range(1, len(weights)+1)
    line = np.poly1d(np.polyfit(xs, weights, 1))
    plt.errorbar(range(1, 13), weights, label='weights',
                 yerr=5, fmt='o', capthick=2, capsize=10)
    plt.plot (xs, line(xs), c='r', label='hypothesis')
    plt.xlim(0, 13); plt.ylim(145, 185)
    plt.xlabel('day')
    plt.ylabel('weight (lbs)')
    book_plots.show_legend()
    plt.show()

def plot_g_h_results(measurements, filtered_data,
                     title='', z_label='Measurements', **kwargs):
    book_plots.plot_filter(filtered_data, **kwargs)
    book_plots.plot_measurements(measurements, label=z_label)
    book_plots.show_legend()
    plt.title(title)
    plt.gca().set_xlim(left=0,right=len(measurements))

if __name__ == '__main__':
    create_predict_update_chart()