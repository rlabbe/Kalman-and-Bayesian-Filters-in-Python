rmdir /s/q html
mkdir html
ipython build_html_ipynb.py
cd html

jupyter nbconvert table_of_contents.ipynb
jupyter nbconvert 00-Preface.ipynb
jupyter nbconvert 01-g-h-filter.ipynb
jupyter nbconvert 02-Discrete-Bayes.ipynb
jupyter nbconvert 03-Gaussians.ipynb
jupyter nbconvert 04-One-Dimensional-Kalman-Filters.ipynb
jupyter nbconvert 05-Multivariate-Gaussians.ipynb
jupyter nbconvert 06-Multivariate-Kalman-Filters.ipynb
jupyter nbconvert 07-Kalman-Filter-Math.ipynb
jupyter nbconvert 08-Designing-Kalman-Filters.ipynb
jupyter nbconvert 09-Nonlinear-Filtering.ipynb
jupyter nbconvert 10-Unscented-Kalman-Filter.ipynb
jupyter nbconvert 11-Extended-Kalman-Filters.ipynb
jupyter nbconvert 12-Particle-Filters.ipynb
jupyter nbconvert 13-Smoothing.ipynb
jupyter nbconvert 14-Adaptive-Filtering.ipynb
jupyter nbconvert Appendix-A-Installation.ipynb
jupyter nbconvert Appendix-B-Symbols-and-Notations.ipynb


