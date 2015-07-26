# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:13:23 2015

@author: rlabbe
"""

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
    predict_template = 'PREDICT: {: 6.2f} {: 6.2f}'
    update_template = 'UPDATE: {: 6.2f} {: 6.2f}\tZ: {:.2f}'

    print(predict_template.format(predict[0], predict[1]),end='\t')
    print(update_template.format(update[0], update[1], z))


def print_variance(positions):
    print('Variance:')
    for i in range(0, len(positions), 5):
        print('\t{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                *[v[1] for v in positions[i:i+5]]))