mkdir tmp
copy ..\*.ipynb .\tmp
copy ..\*.py .\tmp
cp -r ..\kf_book\ .\tmp\

cd tmp

forfiles /m *.ipynb /c "cmd /c ipython ..\rm_notebook.py @file"
jupyter nbconvert --allow-errors --inplace --execute *.ipynb
