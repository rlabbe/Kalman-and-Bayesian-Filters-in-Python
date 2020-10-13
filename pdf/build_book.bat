call run_notebooks
cd ..
ipython merge_book.py

#jupyter nbconvert --to latex --template book book.ipynb
#ipython to_pdf.py
#move /Y book.pdf ../Kalman_and_Bayesian_Filters_in_Python.pdf


rem This is not currently working on my machine. For now I have to do this:
rem $ jupyter nbconvert --to pdf --template book book.ipynb
rem open book.tex in texworks editor.
rem run lualatex conversion twice. Second time creates the TOC.
rem move /Y book.pdf ../Kalman_and_Bayesian_Filters_in_Python.pdf
