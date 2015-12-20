python merge_book.py

ipython nbconvert --to latex --template book book.ipynb
ipython to_pdf.py
move /Y book.pdf ../Kalman_and_Bayesian_Filters_in_Python.pdf
