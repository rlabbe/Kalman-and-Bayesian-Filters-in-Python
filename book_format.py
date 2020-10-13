# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.


Code supporting the book

Kalman and Bayesian Filters in Python
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


This is licensed under an MIT license. See the LICENSE.txt file
for more information.
"""

from contextlib import contextmanager
from IPython.core.display import HTML
import json
import os.path
import sys
import warnings
from kf_book.book_plots import set_figsize, reset_figsize

def test_installation():
    try:
        import filterpy
    except:
        print("Please install FilterPy from the command line by running the command\n\t$ pip install filterpy\n\nSee chapter 0 for instructions.")

    try:
        import numpy
    except:
        print("Please install NumPy before continuing. See chapter 0 for instructions.")

    try:
        import scipy
    except:
        print("Please install SciPy before continuing. See chapter 0 for instructions.")

    try:
        import sympy
    except:
        print("Please install SymPy before continuing. See chapter 0 for instructions.")

    try:
        import matplotlib
    except:
        print("Please install matplotlib before continuing. See chapter 0 for instructions.")

    from distutils.version import LooseVersion

    v = filterpy.__version__
    min_version = "1.4.4"
    if LooseVersion(v) < LooseVersion(min_version):
       print("Minimum FilterPy version supported is {}. "
             "Please install a more recent version.\n"
             "   ex: pip install filterpy --upgrade".format(
             min_version))


    v = matplotlib.__version__
    min_version = "3.0" # this is really old!!!
    if LooseVersion(v) < LooseVersion(min_version):
       print("Minimum Matplotlib version supported is {}. "
             "Please install a more recent version.".format(min_version))

    # require Python 3.6+
    import sys
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 6):
        print('You must use Python version 3.6 or later for the notebooks to work correctly')


    # need to add test for IPython. I think I want to be at 6, which also implies
    # Python 3, matplotlib 2+, etc.

# ensure that we have the correct packages loaded. This is
# called when this module is imported at the top of each book
# chapter so the reader can see that they need to update their environment.
test_installation()


# now that we've tested the existence of all packages go ahead and import

import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np


try:
    matplotlib.style.use('default')
except:
    pass


with warnings.catch_warnings():
    warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)
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


def set_style():
    version = [int(version_no) for version_no in matplotlib.__version__.split('.')]

    try:
        if sys.version_info[0] >= 3:
            style = json.load(open("./kf_book/538.json"))
        else:
            style = json.load(open(".//kf_book/538.json"), object_hook=_decode_dict)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)
            plt.rcParams.update(style)
    except:
        pass
    np.set_printoptions(suppress=True, precision=3,
                        threshold=10000., linewidth=70,
                        formatter={'float':lambda x:' {:.3}'.format(x)})

    # I don't know why I have to do this, but I have to call
    # with suppress a second time or the notebook doesn't suppress
    # exponents
    np.set_printoptions(suppress=True)
    reset_figsize()

    style = '''
        <style>
        .output_wrapper, .output {
            height:auto !important;
            max-height:100000px;
        }
        .output_scroll {
            box-shadow:none !important;
            webkit-box-shadow:none !important;
        }
        </style>
    '''
    jscript = '''
        %%javascript
        IPython.OutputArea.auto_scroll_threshold = 9999;
    '''
    return HTML(style)
