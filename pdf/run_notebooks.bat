mkdir tmp
copy ..\*.ipynb .\tmp
copy ..\*.py .\tmp
cp -r ..\kf_book\ .\tmp\kf_book\

cd tmp

forfiles /m *.ipynb /c "cmd /c ipython ..\rm_notebook.py @file"
forfiles /m *.ipynb /c "cmd /c jupyter nbconvert --to notebook --execute @file --output @file"

