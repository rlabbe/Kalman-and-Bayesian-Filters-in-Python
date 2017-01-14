mkdir tmp
REM copy ..\*.ipynb .\tmp
REM copy ..\*.py .\tmp
REM cp -r ..\code\ .\tmp\code\

cd tmp

REM forfiles /m *.ipynb /c "cmd /c ipython ..\rm_notebook.py @file"
REM forfiles /m *.ipynb /c "cmd /c jupyter nbconvert --to notebook --execute @file --output @file"

