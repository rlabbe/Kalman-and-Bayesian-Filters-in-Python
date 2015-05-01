import IPython.nbconvert.exporters.pdf as pdf

f = open('book.tex', 'r',  encoding="iso-8859-1")
filedata = f.read()
f.close()

newdata = filedata.replace('\chapter{Preface}', '\chapter*{Preface}')

f = open('book.tex', 'w', encoding="iso-8859-1")
f.write(newdata)
f.close()

p = pdf.PDFExporter()
p.run_latex('book.tex')

