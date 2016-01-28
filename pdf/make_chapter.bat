cp %1 chapter.ipynb
ipython nbconvert --to latex chapter.ipynb
ipython to_pdf.py chapter.tex


