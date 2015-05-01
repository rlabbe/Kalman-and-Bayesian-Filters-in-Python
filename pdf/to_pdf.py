from __future__ import print_function
import io

import IPython.nbconvert.exporters.pdf as pdf
import fileinput

'''
for line in fileinput.input('book.tex', openhook=fileinput.hook_encoded("iso-8859-1")):
    #print(line.replace('\chapter{Preface}\label{preface}', '\chapter*{Preface}\label{preface}'), end='')
#    line.replace('    \chapter{Preface}\label{preface}', '    \chapter*{Preface}\label{preface}')
    line.replace('shit', 'poop')
'''

f = open('book.tex', 'r',  encoding="iso-8859-1")
filedata = f.read()
f.close()

newdata = filedata.replace('\chapter{Preface}', '\chapter*{Preface}')

f = open('book.tex', 'w',  encoding="iso-8859-1")
f.write(newdata)
f.close()

p = pdf.PDFExporter()
p.run_latex('book.tex')

