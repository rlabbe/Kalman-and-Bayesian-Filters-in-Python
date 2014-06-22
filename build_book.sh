#! /bin/bash

echo "merging book..."

ipython nbmerge.py Preface.ipynb Signals_and_Noise.ipynb g-h_filter.ipynb discrete_bayes.ipynb Gaussians.ipynb Kalman_Filters.ipynb Multidimensional_Kalman_Filters.ipynb Kalman_Filter_Math.ipynb Designing_Kalman_Filters.ipynb Extended_Kalman_Filters.ipynb Unscented_Kalman_Filter.ipynb > Kalman_and_Bayesian_Filters_in_Python.ipynb

echo "creating pdf..."
ipython nbconvert --to latex --template book --post PDF Kalman_and_Bayesian_Filters_in_Python.ipynb

echo "done."

