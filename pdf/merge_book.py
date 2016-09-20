from __future__ import print_function
import io
import nbformat
import sys
from formatting import *


def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename, encoding='utf-8') as f:
        s = f.read()
        if old_string not in s:
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w', encoding='utf-8') as f:
        s = s.replace(old_string, new_string)
        f.write(s)


def merge_notebooks(outfile, filenames):
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

    outfile.write(nbformat.writes(merged, nbformat.NO_CONVERT))


if __name__ == '__main__':
    with open('book.ipynb', 'w', encoding='utf-8') as f:
        '''merge_notebooks(f,
            ['../14-Adaptive-Filtering.ipynb'])'''

        merge_notebooks(f,
            ['../00-Preface.ipynb',
             '../01-g-h-filter.ipynb',
             '../02-Discrete-Bayes.ipynb',
             '../03-Gaussians.ipynb',
             '../04-One-Dimensional-Kalman-Filters.ipynb',
             '../05-Multivariate-Gaussians.ipynb',
             '../06-Multivariate-Kalman-Filters.ipynb',
             '../07-Kalman-Filter-Math.ipynb',
             '../08-Designing-Kalman-Filters.ipynb',
             '../09-Nonlinear-Filtering.ipynb',
             '../10-Unscented-Kalman-Filter.ipynb',
             '../11-Extended-Kalman-Filters.ipynb',
             '../12-Particle-Filters.ipynb',
             '../13-Smoothing.ipynb',
             '../14-Adaptive-Filtering.ipynb',
             '../Appendix-A-Installation.ipynb',
             '../Appendix-B-Symbols-and-Notations.ipynb',
             '../Appendix-D-HInfinity-Filters.ipynb',
             '../Appendix-E-Ensemble-Kalman-Filters.ipynb'])


    #remove text printed for matplotlib charts
    inplace_change('book.ipynb', '<IPython.core.display.Javascript object>', '')
    inplace_change('book.ipynb', '<IPython.core.display.HTML object>', '')