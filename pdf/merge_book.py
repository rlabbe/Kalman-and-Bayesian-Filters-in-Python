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
            #remove_formatting(nb)
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

        merge_notebooks(f,
            ['./tmp/00-Preface.ipynb',
             './tmp/01-g-h-filter.ipynb',
             './tmp/02-Discrete-Bayes.ipynb',
             './tmp/03-Gaussians.ipynb',
             './tmp/04-One-Dimensional-Kalman-Filters.ipynb',
             './tmp/05-Multivariate-Gaussians.ipynb',
             './tmp/06-Multivariate-Kalman-Filters.ipynb',
             './tmp/07-Kalman-Filter-Math.ipynb',
             './tmp/08-Designing-Kalman-Filters.ipynb',
             './tmp/09-Nonlinear-Filtering.ipynb',
             './tmp/10-Unscented-Kalman-Filter.ipynb',
             './tmp/11-Extended-Kalman-Filters.ipynb',
             './tmp/12-Particle-Filters.ipynb',
             './tmp/13-Smoothing.ipynb',
             './tmp/14-Adaptive-Filtering.ipynb',
             './tmp/Appendix-A-Installation.ipynb',
             './tmp/Appendix-B-Symbols-and-Notations.ipynb',
             './tmp/Appendix-D-HInfinity-Filters.ipynb',
             './tmp/Appendix-E-Ensemble-Kalman-Filters.ipynb'])


    #remove text printed for matplotlib charts
    inplace_change('book.ipynb', '<IPython.core.display.Javascript object>', '')
    inplace_change('book.ipynb', '<IPython.core.display.HTML object>', '')