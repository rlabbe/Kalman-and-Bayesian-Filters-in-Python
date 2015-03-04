from __future__ import print_function
import io

import IPython.nbconvert.exporters.pdf as pdf
import fileinput

for line in fileinput.input('Kalman_and_Bayesian_Filters_in_Python.tex', inplace=True):
    print(line.replace('\chapter{Preface}', '\chapter*{Preface}'), end='')


p = pdf.PDFExporter()
p.run_latex('Kalman_and_Bayesian_Filters_in_Python.tex')

