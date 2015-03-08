
from __future__ import print_function
import IPython.nbformat as nbformat

from formatting import *


def prep_for_html_conversion(filename):
    added_appendix = False
    with io.open('../'+filename, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)
        remove_formatting(nb)
        if not added_appendix and filename[0:8] == 'Appendix':
            remove_links_add_appendix(nb)
            added_appendix = True
        else:
            remove_links(nb)
        nbformat.write(nb, filename)
 

if __name__ == '__main__':
    notebooks = \
        ['00_Preface.ipynb',
         '01_g-h_filter.ipynb',
         'Appendix_A_Installation.ipynb']
         
    for n in notebooks:
        prep_for_html_conversion(n)

    '''merge_notebooks(
        ['../00_Preface.ipynb',
         '../01_g-h_filter.ipynb',
         '../02_Discrete_Bayes.ipynb',
         '../03_Least_Squares_Filters.ipynb',
         '../04_Gaussians.ipynb',
         '../05_Kalman_Filters.ipynb',
         '../06_Multivariate_Kalman_Filters.ipynb',
         '../07_Kalman_Filter_Math.ipynb',
         '../08_Designing_Kalman_Filters.ipynb',
         '../09_Nonlinear_Filtering.ipynb',
         '../10_Unscented_Kalman_Filter.ipynb',
         '../11_Extended_Kalman_Filters.ipynb',
         '../12_Designing_Nonlinear_Kalman_Filters.ipynb',
         '../13_Smoothing.ipynb',
         '../14_Adaptive_Filtering.ipynb',
         '../15_HInfinity_Filters.ipynb',
         '../16_Ensemble_Kalman_Filters.ipynb',
         '../Appendix_A_Installation.ipynb',
         '../Appendix_B_Symbols_and_Notations.ipynb'])'''
