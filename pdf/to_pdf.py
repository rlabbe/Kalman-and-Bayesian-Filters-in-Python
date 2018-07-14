import nbconvert.exporters.pdf as pdf
import sys

if len(sys.argv) == 2:
    name = sys.argv[1]
else:
    name = 'book.tex'
    
f = open(name, 'r',  encoding="iso-8859-1")
filedata = f.read()
f.close()

newdata = filedata.replace('\chapter{Preface}', '\chapter*{Preface}')

f = open(name, 'w', encoding="iso-8859-1")
f.write(newdata)
f.close()

p = pdf.PDFExporter()
p.run_latex(name)

