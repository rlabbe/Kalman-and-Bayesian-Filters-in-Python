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

from matplotlib import animation
import matplotlib.pyplot as plt


    
def animate(filename, func, frames, interval, fig=None, figsize=(6.5, 6.5)):
    """ Creates an animated GIF of a matplotlib.

    Parameters
    ----------
    filename : string
        name of the file. E.g 'foo.GIF' or '\home\monty\parrots\fjords.gif'

    func : function
       function that will be called once per frame. Must have signature of
       def fun_name(frame_num)

    frames : int
       number of frames to animate. The current frame number will be passed
       into func at each call.

    interval : float
       Milliseconds to pause on each frame in the animation. E.g. 500 for half
       a second.

    figsize : (float, float)  optional
       size of the figure in inches. Defaults to 6.5" by 6.5"
    """

    def init_func():
        """ This draws the 'blank' frame of the video. To work around a bug
        in matplotlib 1.5.1 (#5399) you must supply an empty init function
        for the save to work."""
        
        pass

    if fig is None:
        fig = plt.figure(figsize=figsize)

    anim = animation.FuncAnimation(fig, func, init_func=init_func,
        frames=frames, interval=interval)
    anim.save(filename, writer='imagemagick')