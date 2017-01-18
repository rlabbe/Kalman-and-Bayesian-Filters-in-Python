REM WINDOWS script to delete all files

rm --f *.tex
rm --f *.toc
rm --f *.aux
rm --f *.log
rm --f *.out
rm --f book.ipynb
rm --f book.toc
rm --f book.tex

rm --f chapter.ipynb
rm --f chapter.pdf

rmdir  /S /Q book_files
rmdir  /S /Q chapter_files
rmdir  /S /Q tmp
