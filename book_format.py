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

from contextlib import contextmanager
from IPython.core.display import HTML
import json
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import warnings
from kf_book.book_plots import set_figsize, reset_axis

# version 1.4.3 of matplotlib has a bug that makes
# it issue a spurious warning on every plot that
# clutters the notebook output
if matplotlib.__version__ == '1.4.3':
    warnings.simplefilter(action="ignore", category=FutureWarning)

np.set_printoptions(precision=3)

def test_filterpy_version():

    import filterpy
    from distutils.version import LooseVersion

    v = filterpy.__version__
    min_version = "0.1.2"
    if LooseVersion(v) < LooseVersion(min_version):
       raise Exception("Minimum FilterPy version supported is {}.\n"
                       "Please install a more recent version.\n"
                       "   ex: pip install filterpy --upgrade".format(
             min_version))


# ensure that we have the correct filterpy loaded. This is
# called when this module is imported at the top of each book
# chapter so the reader can see that they need to update FilterPy.
test_filterpy_version()

pylab.rcParams['figure.max_open_warning'] = 50




@contextmanager
def numpy_precision(precision):
	old = np.get_printoptions()['precision']
	np.set_printoptions(precision=precision)
	yield
	np.set_printoptions(old)

@contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

def _decode_list(data):
    rv = []
    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv

def _decode_dict(data):
    rv = {}
    for key, value in data.iteritems():
        if isinstance(key, unicode):
            key = key.encode('utf-8')
        if isinstance(value, unicode):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = _decode_list(value)
        elif isinstance(value, dict):
            value = _decode_dict(value)
        rv[key] = value
    return rv


def load_style(directory = '.', name='kf_book/custom.css'):
        version = [int(version_no) for version_no in matplotlib.__version__.split('.')]

        try:
            if sys.version_info[0] >= 3:
                style = json.load(open(os.path.join(directory, "kf_book/538.json")))
            else:
                style = json.load(open(directory + "/kf_book/538.json"), object_hook=_decode_dict)

            # matplotlib has deprecated the use of axes.color_cycle as of version
            if version[0] > 1 or (version[0] == 1 and version[1] >= 5):
                style["axes.prop_cycle"] = "cycler('color', ['#6d904f','#013afe', '#202020','#fc4f30','#e5ae38','#A60628','#30a2da','#008080','#7A68A6','#CF4457','#188487','#E24A33'])"
                style.pop("axes.color_cycle", None)
            plt.rcParams.update(style)
        except:
            pass
        set_figsize()
        reset_axis ()
        np.set_printoptions(suppress=True,precision=3, linewidth=70,
                            formatter={'float':lambda x:' {:.3}'.format(x)})

        styles = open(os.path.join(directory, name), 'r').read()
        set_figsize()
        return HTML(styles)
