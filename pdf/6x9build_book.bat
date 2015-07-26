REM copy /Y  ..\14-Adaptive-Filtering.ipynb book.ipynb
python merge_book.py

ipython nbconvert --to latex --template book6x9 book.ipynb
ipython to_pdf.py
REM move /Y book.pdf book6x9.pdf
