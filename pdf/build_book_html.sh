#! /bin/bash

echo "merging book..."


echo "creating html..."
ipython nbconvert --to=html table_of_contents.ipynb
ipython nbconvert --to=html Preface.ipynb
ipython nbconvert --to=html 01_gh_filter/g-h_filter.ipynb


echo "done."

