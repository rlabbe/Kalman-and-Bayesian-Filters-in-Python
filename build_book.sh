#! /bin/bash

echo "merging book..."

python merge_book.py > Kalman_and_Bayesian_Filters_in_Python.ipynb

echo "creating pdf..."
ipython nbconvert --to latex --template book --post PDF Kalman_and_Bayesian_Filters_in_Python.ipynb

echo "done."

