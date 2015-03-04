from __future__ import print_function
import io
import IPython.nbformat as nbformat
import sys


def remove_formatting(nb):
    c = nb['cells']
    for i in range (len(c)):
        if 'source' in c[i].keys():
            if c[i]['source'][0:16] == '#format the book':
                del c[i]
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
    #merge_notebooks(sys.argv[1:])
    merge_notebooks(
        ['../00_Preface.ipynb',
         '../01_g-h_filter.ipynb',
         '../Appendix_A_Installation.ipynb'])
