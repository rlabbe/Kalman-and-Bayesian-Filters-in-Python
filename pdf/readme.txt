This directory contains code to convert the book into the PDF file. The normal
build process is to cd into this directory, and run build_book from the command
line. If the build is successful (no errors printed), then run clean_book from
the command line. clean_book is not run automatically because if there is an
error you probably need to look at the intermediate output to debug the issue.

I build the PDF my merging all of the notebooks into one huge one. I strip out
the initial cells for the book formatting and table of contents, and do a few
other things so it renders well in PDF.

I used to do this in Unix, but switched to Windows. The Unix scripts have not
been kept up to date.

There is also some experimental code to convert to html.

The files with short in the name combine only a couple of notebooks together.
I use this to test the production without having to wait the relatively long
time required to produce the entire book. Mostly this is for testing the
scripts.

No one but me should need to run this stuff, but if you fork the project and
want to generate a PDF, this is how you do it.
