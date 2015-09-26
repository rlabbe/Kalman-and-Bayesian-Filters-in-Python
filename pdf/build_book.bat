python merge_book.py

ipython nbconvert --to latex --template book6x9 book.ipynb
ipython to_pdf.py
move /Y book.pdf ../book6x9.pdf

ipython nbconvert --to latex --template book book.ipynb
ipython to_pdf.py
move /Y book.pdf ../Kalman_and_Bayesian_Filters_in_Python.pdf
