# -*- coding: utf-8 -*-
from IPython.core.display import HTML
import matplotlib.pylab as pylab

def load_style(name='../styles/custom2.css'):
    styles = open(name, 'r').read()
    return HTML(styles)

pylab.rcParams['lines.linewidth'] = 2
pylab.rcParams['lines.antialiased'] = True
pylab.rcParams['patch.linewidth'] = 0.5
pylab.rcParams['patch.facecolor'] = '348ABD' #blue
pylab.rcParams['patch.edgecolor'] = 'eeeeee'
pylab.rcParams['patch.antialiased'] = True
pylab.rcParams['font.family'] = 'monospace'
pylab.rcParams['font.size'] = 12.0
#pylab.rcParams['font.monospace'] = 'Andale Mono, Nimbus Mono L, Courier New, Courier, Fixed, Terminal, monospace'
pylab.rcParams['axes.facecolor'] = 'E5E5E5'
pylab.rcParams['axes.edgecolor'] = 'bcbcbc'
pylab.rcParams['axes.linewidth'] = 1
pylab.rcParams['axes.grid'] = True
pylab.rcParams['axes.titlesize'] = 'x-large'
pylab.rcParams['axes.labelsize'] = 'large'
pylab.rcParams['axes.labelcolor'] = '555555'
pylab.rcParams['axes.axisbelow'] = True
pylab.rcParams['axes.color_cycle'] = '004080, 8EBA42, E24A33, 348ABD, 777760, 988ED5, FDC15E'
pylab.rcParams['xtick.major.pad'] = 6
pylab.rcParams['xtick.minor.size'] = 0
pylab.rcParams['xtick.minor.pad'] = 6
pylab.rcParams['xtick.color'] = '555555'
pylab.rcParams['ytick.direction'] = 'in'
pylab.rcParams['legend.fancybox'] = True
pylab.rcParams['figure.facecolor'] = '0.85'
pylab.rcParams['figure.edgecolor'] = '0.50'
pylab.rcParams['figure.subplot.hspace'] = 0.5
pylab.rcParams['figure.figsize'] = 12,6
pylab.rcParams['grid.color'] = 'ffffff'
pylab.rcParams['grid.linestyle'] = 'solid'
pylab.rcParams['grid.linewidth'] = 1.5


