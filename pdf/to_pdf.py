from __future__ import print_function
import io

import IPython.nbconvert.exporters.pdf as pdf
import fileinput

for line in fileinput.input('book.tex', openhook=fileinput.hook_encoded("iso-8859-1")):
#    print(line.replace('\chapter{Preface}', '\chapter*{Preface}'), end='')
    line.replace('\chapter{Preface}', '\chapter*{Preface}')


p = pdf.PDFExporter()
p.run_latex('book.tex')

