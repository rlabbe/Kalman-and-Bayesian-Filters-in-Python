#! /bin/bash

git checkout gh-pages
git checkout master Kalman_and_Bayesian_Filters_in_Python.pdf
git add Kalman_and_Bayesian_Filters_in_Python.pdf
git commit -m 'updating PDF'
git push
git checkout master