copy /Y  ..\01-g-h-filter.ipynb book.ipynb

ipython nbconvert --to latex --template book6x9 book.ipynb
ipython to_pdf.py
REM move /Y book.pdf book6x9.pdf
