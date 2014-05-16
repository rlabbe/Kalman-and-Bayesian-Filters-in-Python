#[Kalman Filters and Random Signals in Python](http://github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python)

### Version 0.0 - not ready for public consumption. In development.

The motivation for this book came out of my desire for a gentle introduction to Kalman filtering. I'm a software engineer that spent almost two decades in the avionics field, and so I have always been 'bumping elbows' with the Kalman filter, but never had the need to implement one myself. As I moved into solving tracking problems with computer vision I needed to start implementing them for myself. There are classic textbooks in the field, such as Grewal and Andrew's excellent *Kalman Filtering*. But sitting down and trying to read these books is a dismaying experience. I am in no way putting down the book, but this is the opening to chapter one.
> Theoretically, it is an estimator for what is called **the linear-quadratic-Gaussian problem**, which is the problem of estimating the instantaneous "state" ... of a linear dynamic system perturbed by Gausing white noise - by using measurements linearly related to the state, but corrupted by the Gaussian white noise. The resulting estimator is statistically optimal with respect to any quadratic function of estimation error.

That is actually a quite approachable sentence, so long as you know what all the terms mean. But the next few chapters fly through several years of undergraduate math, blithely referring you to textbooks on, for example, Itō calculus, and presenting an entire semester's worth of statistics in a few brief paragraphs. It is an excellent resource for an upper undergraduate course, and I can now read it, but sans instructor and resources it is heavy going. 

However, as I began to finally understand the Kalman filter I realized the underlying concepts are quite straightforward. A few simple probability rules, some intuition about how we integrate disparate knowledge to explain events in our everyday life and the core concepts of the Kalman filter are accessible.

As I began to understand the math and theory another difficulty presented itself. A book or paper's author makes some statement of fact and presents a graph as proof. Well, why the statement is true is not clear, nor is the method by which you might make that plot obvious. Or maybe I wonder "is this true if $R=0$?" 


Contents
-----

* [**Introduction**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python/master/Introduction.ipynb)
 
    Introduction to the Kalman filter. Explanation of the idea behind this book.


* [**Chapter 1: The g-h Filter**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python/master/g-h_filter.ipynb)

    Intuitive introduction to the g-h filter, which is a family of filters that includes the Kalman filter. Not filler - once you understand this chapter you will understand the concepts behind the Kalman filter. 


* [**Chapter 2: The Discrete Bayes Filter**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python/master/histogram_filter.ipynb)
    Introduces the Discrete Bayes Filter. From this you will learn the probabilistic reasoning that underpins the Kalman filter in an easy to digest form.

* [**Chapter 3: Gaussian Probabilities**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python/master/Gaussians.ipynb)
    Introduces using Gaussians to represent beliefs. Gaussians allow us to implement the algorithms used in the Discrete Bayes Filter to work in continuous domains.

* [**Chapter 4: One Dimensional Kalman Filters**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python/master/Kalman_Filters.ipynb)
    Implements a Kalman filter by modifying the Discrete Bayesian Filter to use Gaussians. This is a full featured Kalman filter, albeit only useful for 1D problems. 

* [**Chapter 5: Multidimensional Kalman Filter**](http://nbviewer.ipython.org/urls/raw.github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python/master/Multidimensional_Kalman_Filters.ipynb)
    We extend the Kalman filter developed in the previous chapter to the full, generalized filter. 


Reading the book
-----


Installation
-----

License
-----

Contact
-----
