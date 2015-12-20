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

import book_plots as bp
import matplotlib.pyplot as plt

def plot_dog_track(xs, measurement_var, process_var):
    N = len(xs)
    bp.plot_track([0, N-1], [1, N])
    bp.plot_measurements(xs, label='Sensor')
    bp.set_labels('variance = {}, process variance = {}'.format(
              measurement_var, process_var), 'time', 'pos')
    plt.ylim([0, N])
    bp.show_legend()
    plt.show()


def print_gh(predict, update, z):
    predict_template = '         {: 7.3f} {: 8.3f}'
    update_template = '{: 7.3f} {: 7.3f}\t   {:.3f}'

    print(predict_template.format(predict[0], predict[1]),end='\t')
    print(update_template.format(update[0], update[1], z))


def print_variance(positions):
    print('Variance:')
    for i in range(0, len(positions), 5):
        print('\t{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                *[v[1] for v in positions[i:i+5]]))