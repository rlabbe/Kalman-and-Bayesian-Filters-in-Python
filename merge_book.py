from __future__ import print_function
import io
from IPython.nbformat import current
import sys

def remove_formatting(nb):
    w = nb['worksheets']
    node = w[0]
    c = node['cells']
    for i in range (len(c)):
        if 'input' in c[i].keys():
            if c[i]['input'][0:16] == '#format the book':
                del c[i]
                return


def merge_notebooks(filenames):
    merged = None
    for fname in filenames:
        with io.open(fname, 'r', encoding='utf-8') as f:
            nb = current.read(f, u'json')
            remove_formatting(nb)
        if merged is None:
            merged = nb
        else:
            merged.worksheets[0].cells.extend(nb.worksheets[0].cells)
    merged.metadata.name += "_merged"

    print(current.writes(merged, u'json'))


if __name__ == '__main__':
    #merge_notebooks(sys.argv[1:])
    merge_notebooks(
        ['Preface.ipynb',
         'g-h_filter.ipynb',
         'discrete_bayes.ipynb',
         'Least_Squares_Filters.ipynb',
         'Gaussians.ipynb',
         'Kalman_Filters.ipynb',
         'Multivariate_Kalman_Filters.ipynb',
         'Kalman_Filter_Math.ipynb',
         'Designing_Kalman_Filters.ipynb',
         'Extended_Kalman_Filters.ipynb',
         'Unscented_Kalman_Filter.ipynb',
         'Designing_Nonlinear_Kalman_Filters.ipynb',
         'Appendix_Installation.ipynb',
         'Appendix_Symbols_and_Notations.ipynb'])


#    merge_notebooks(['Preface.ipynb', g-h_filter.ipynb discrete_bayes.ipynb Gaussians.ipynb Kalman_Filters.ipynb Multivariate_Kalman_Filters.ipynb Kalman_Filter_Math.ipynb Designing_Kalman_Filters.ipynb Extended_Kalman_Filters.ipynb Unscented_Kalman_Filter.ipynb'])
