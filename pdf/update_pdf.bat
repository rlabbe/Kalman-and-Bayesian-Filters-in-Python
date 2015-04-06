cd ..
git checkout gh-pages
git pull
git checkout master Kalman_and_Bayesian_Filters_in_Python.pdf
git checkout master README.md
git add Kalman_and_Bayesian_Filters_in_Python.pdf
git add README.md
git commit -m "updating PDF"
git push
git checkout master
cd pdf
