# -*- coding: utf-8 -*-
from IPython.core.display import HTML
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import json

def test_filterpy_version():
    import filterpy
    min_version = [0,0,10]
    v = filterpy.__version__
    tokens = v.split('.')
    for i,v in enumerate(tokens):
        if int(v) > min_version[i]:
            return

    i = len(tokens) - 1
    if min_version[i] > int(tokens[i]):
       raise Exception("Minimum FilterPy version supported is {}.{}.{}.\n"
                       "Please install a more recent version." .format(
             *min_version))
    v = int(tokens[0]*1000)

def load_style(name='../styles/custom2.css'):
    styles = open(name, 'r').read()
    return HTML(styles)


# ensure that we have the correct filterpy loaded. This is
# called when this module is imported at the top of each book
# chapter so the reader can see that they need to update FilterPy.
test_filterpy_version()


def equal_axis():
    pylab.rcParams['figure.figsize'] = 10,10
    plt.axis('equal')

def reset_axis():
    pylab.rcParams['figure.figsize'] = 12, 6


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

import sys
if sys.version_info[0] >= 3:
    s = json.load( open("../code/538.json"))
else:
    s = json.load( open("../code/538.json"), object_hook=_decode_dict)
plt.rcParams.update(s)
reset_axis ()
