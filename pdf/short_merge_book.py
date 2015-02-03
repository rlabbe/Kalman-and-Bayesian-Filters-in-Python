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

def remove_links(nb):
    w = nb['worksheets']
    node = w[0]
    c = node['cells']
    for i in range (len(c)):
        if 'source' in c[i].keys():
            if c[i]['source'][0:19] == '[Table of Contents]':
                del c[i]
                return

def remove_links_add_appendix(nb):
    w = nb['worksheets']
    node = w[0]
    c = node['cells']
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
            nb = current.read(f, u'json')
            remove_formatting(nb)
            if not added_appendix and fname[0:8] == 'Appendix':
                remove_links_add_appendix(nb)
                added_appendix = True
            else:
                remove_links(nb)
        if merged is None:
            merged = nb
        else:
            merged.worksheets[0].cells.extend(nb.worksheets[0].cells)
    merged.metadata.name += "_merged"

    print(current.writes(merged, u'json'))


if __name__ == '__main__':
    #merge_notebooks(sys.argv[1:])
    merge_notebooks(
        ['../01_g-h_filter.ipynb',
         '../02_Discrete_Bayes.ipynb'])
