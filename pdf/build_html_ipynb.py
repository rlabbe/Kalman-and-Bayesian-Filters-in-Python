
from __future__ import print_function
import IPython.nbformat as nbformat

from formatting import *
from os.path import join

def prep_for_html_conversion(filename):
    added_appendix = False
    with io.open(join('..', filename), 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)
        remove_formatting(nb)
        if not added_appendix and filename[0:8] == 'Appendix':
            remove_links_add_appendix(nb)
            added_appendix = True
        else:
            remove_links(nb)
        nbformat.write(nb, join('html', filename))
 

if __name__ == '__main__':
    notebooks = \
            ['table_of_contents.ipynb',
             '00-Preface.ipynb',
             '01-g-h-filter.ipynb',
             '02-Discrete-Bayes.ipynb',
             '03-Gaussians.ipynb',
             '04-One-Dimensional-Kalman-Filters.ipynb',
             '05-Multivariate-Gaussians.ipynb',
             '06-Multivariate-Kalman-Filters.ipynb',
             '07-Kalman-Filter-Math.ipynb',
             '08-Designing-Kalman-Filters.ipynb',
             '09-Nonlinear-Filtering.ipynb',
             '10-Unscented-Kalman-Filter.ipynb',
             '11-Extended-Kalman-Filters.ipynb',
             '12-Particle-Filters.ipynb',
             '13-Smoothing.ipynb',
             '14-Adaptive-Filtering.ipynb',
             'Appendix-A-Installation.ipynb',
             'Appendix-B-Symbols-and-Notations.ipynb',
             'Appendix-D-HInfinity-Filters.ipynb',
             'Appendix-E-Ensemble-Kalman-Filters.ipynb']
         
    for n in notebooks:
        prep_for_html_conversion(n)