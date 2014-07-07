#! /bin/bash
rm --f *.tex
rm --f *.toc
rm --f ./*_files/*.png
rm --f Kalman_and_Bayesian_Filters_in_Python.ipynb
rm --f Kalman_and_Bayesian_Filters_in_Python.toc
rm --f Kalman_and_Bayesian_Filters_in_Python.tex
rmdir  ./*_files/ 2> /dev/null

if (( $# == 1)); then
  if [ "@1" == all ]; then
    rm Kalman_and_Bayesian_Filters_in_Python.pdf;
  fi 
fi
