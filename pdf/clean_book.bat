REM WINDOWS script to delete all files

rm --f *.tex
rm --f *.toc
rm --f *.aux
rm --f *.log
rm --f *.out

rm --f book.ipynb
rm --f book.toc
rm --f book.tex
rmdir  /S /Q book_files
