from __future__ import print_function
import io
import IPython.nbformat as nbformat
import sys


def remove_formatting(nb):
    cells = nb['cells']
    for i in range (len(cells)):
        if 'source' in cells[i].keys():
            if cells[i]['source'][0:16] == '#format the book':
                del cells[i]
                return


def remove_links(nb):
    c = nb['cells']
    for i in range (len(c)):
        if 'source' in c[i].keys():
            if c[i]['source'][0:19] == '[Table of Contents]':
                del c[i]
                return


def remove_links_add_appendix(nb):
    c = nb['cells']
    for i in range (len(c)):
        if 'source' in c[i].keys():
            if c[i]['source'][0:19] == '[Table of Contents]':
                c[i]['source'] = '\\appendix'
                return


def merge_notebooks(filenames):
    merged = None
    added_appendix = False
    for fname in filenames:
        with io.open(fname, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, nbformat.NO_CONVERT)
            remove_formatting(nb)
            if not added_appendix and fname[0:8] == 'Appendix':
                remove_links_add_appendix(nb)
                added_appendix = True
            else:
                remove_links(nb)
        if merged is None:
            merged = nb
        else:
            merged.cells.extend(nb.cells)
    #merged.metadata.name += "_merged"

    print(nbformat.writes(merged, nbformat.NO_CONVERT))


if __name__ == '__main__':
    '''merge_notebooks(
        ['../00_Preface.ipynb',
         '../01_g-h_filter.ipynb',
         '../Appendix_A_Installation.ipynb'])'''

    merge_notebooks(
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
         '../Appendix_B_Symbols_and_Notations.ipynb'])
