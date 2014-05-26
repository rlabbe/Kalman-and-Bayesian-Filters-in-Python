Read the static version [HERE](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/toc.ipynb)

#[Kalman Filters and Random Signals in Python](http://github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python)

### Version 0.0 - not ready for public consumption. In development.


## Motivation

This is a book for programmers that have a need or interest in Kalman filtering. The motivation for this book came out of my desire for a gentle introduction to Kalman filtering. I'm a software engineer that spent almost two decades in the avionics field, and so I have always been 'bumping elbows' with the Kalman filter, but never had the need to implement one myself. As I moved into solving tracking problems with computer vision I needed to start implementing them. There are classic textbooks in the field, such as Grewal and Andrew's excellent *Kalman Filtering*. But sitting down and trying to read these books is a dismal and trying experience if you do not have the background. Typcially the first few chapters fly through several years of undergraduate math, blithely referring you to textbooks on, for example, Itō calculus, and presenting an entire semester's worth of statistics in a few brief paragraphs. These books are good textbooks for an upper undergraduate course, and an invaluable reference to researchers and professionals, but the going is truly difficult for the more casual reader. Symbology is introduced without explanation, different texts use different words and variables names for the same concept, and the books are almost devoid of examples or worked problems. I often found myself able to parse the words and comprehend the mathematics of a defition, but had no idea as to what real world phenomena these words and math were attempting to describe. "But what does that *mean?*" was my repeated thought.

However, as I began to finally understand the Kalman filter I realized the underlying concepts are quite straightforward. A few simple probability rules, some intuition about how we integrate disparate knowledge to explain events in our everyday life and the core concepts of the Kalman filter are accessible. Kalman filters have a reputation for difficulty, but shorn of much of the formal terminology the beauty of the subject and of their math became clear to me, and I fell in love with the topic. 

As I began to understand the math and theory more difficulties itself. A book or paper's author makes some statement of fact and presents a graph as proof.  Unfortunately, why the statement is true is not clear to me, nor is the method by which you might make that plot obvious. Or maybe I wonder "is this true if R=0?"  Or the author provide pseudocode - at such a high level that the implementation is not obvious. Some books offer Matlab code, but I do not have a license to that expensive package. Finally, many books end each chapter with many useful exercises. Exercises which you need to understand, but excercises with no answers. If you are using the book in a classroom, perhaps this is okay, but it is terrible for the independent reader. I loathe that an author witholds information from me, presumably to avoid 'cheating'. 
None of this necessary, from my point of view. Certainly if you are designing a Kalman filter for a aircraft or missile you must be a thorough master of all of the mathematics and topics in a typical Kalman filter textbook. I just want to track an image on a screen, or write some code for my Arduino project. I want to know how the plots in the book are made, and chose different parameters than the author chose. I want to run simulations. I want to inject more noise in the signal and see how a filter performs. 

I wrote this book to address all of those needs. This is not the book for you if you program avionics for Boeing or design radars for Ratheon. Go get a degree at Georgia Tech, UW, or the like, because you'll need it for that kind of work. This book is for the hobbiest, the curious, and the working engineer that needs to filter or smooth data. 

This book is interactive. While you can read it online as static content, I urge you to use it as intended. It is written using IPython Notebook, which allows you to combine text, python, and python output in one place. Every plot, every piece of data in this book is generated from Python that is available to you right inside the notebook. Want to double the value of a parameter. Click on the Python cell, change the parameter's value, and click 'Run'. A new plot or printed output will appear in the book. 

This book has exercises, but it also has the answers. I trust you. If you just need an answer, go ahead and read the answer. If you want to internalize this knowledge, try to implement the exercise before you read the answer. 

This book has supporting libraries for computing statistics, plotting various things related to filters, and for the various filters that we cover. This does require a strong caveat; most code is written for didactic purposes. It is rare that I chose the most efficient solution (which often obscures the intent of the code), and I mostly did not concern myself with numerical stability. This is important to understand - Kalman filters in aircraft are carefully designed and implemented to be numerically stable; the naive implemention is not stable in many cases. If you are serious about Kalman filters this book will not be the last book you need. My intention is to introduce you to the concepts and mathematics, and to get you to the point where the textbooks are approachable.

Finally, this book is free. The cost for the books required to learn Kalman filtering is somewhat prohibitive even for a Silicon Valley engineer like myself; I cannot believe the are within the reach of someone in a depressed economy, or a financially struggling student. I have gained so much from free software like Python, and free books like those from Allen B. Downey [here](http://www.greenteapress.com/). It's time to repay that. So, the book is free, it is hosted on free servers, and it uses free software for all of the code. 




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
There are multiple ways to read this book. However, it is intended to be interactive and I recommend using it in that form. If you install IPython on your computer and then clone this book you will be able to run all of the code in the book yourself. You can perform experiments, see how filters react to different data, see how different filters react to the same data, and so on. I find this sort of immediate feedback both vital and invigorating. You do not have to wonder "what happens if". Try it and see!

If you do not want to do that you can read this book online. the website [nbviewer]http://nbviewer.org provides an IPython Notebook server that renders a notebook stored at github (or elsewhere). The rendering is done in real time when you load the book. If you read my book today, and then I make a change tomorrow, when you go back tomorrow you will see that change. 

You may access this book via nbviewer at any by using this address:
http://nbviewer.ipython.org/github/rlabbe/Kalman-Filters-and-Random-Signals-in-Python/blob/master/Introduction.ipynb

Finally, you may generate output in a variety of formats. I will not cover how to do that, other than to point you to [IPython nbconvert](http://ipython.org/ipython-doc/rel-1.0.0/interactive/nbconvert.html). You can convert this book into static HTML pages, latex, or PDF. While I don't recommend it particularly, it is useful for those that don't want to program and/or are working offline.


Installation and Software Requirements
-----
If you want to run the notebook on your computer, which is what I recommend, then you will have to have IPython installed. I do not cover how to do that in this book; requirements change based on what other python installations you may have, whether you use a third party package like Anaconda Python, what operating system you are using, and so on. 

To use all features you will have to have Ipython 2.0 installed, which is released and stable as of April 2014. Most of the book does not require that, but I do make use of the interactive plotting widgets introduced in this release. A few cells will not run if you have an older version installed.

You will need Python 2.7 or later installed. Almost all of my work is done in Python 2.7, but I periodically test on 3.3. I do not promise any specific check in will work in 3.X, however. I do use Python's "from __future__ import ..." statement to help with compatibility. For example, all prints need to use parenthesis. If you try to add, say, "print 3.14" into the book your script will fail; you must write "print (3.4)" as in Python 3.X.

You will need a recent version of NumPy, SciPy, and Matplotlib installed. I don't really know what the minimal might be. 
I have numpy 1.71, SciPy 0.13.0, and Matplotlib 1.3.1 installed on my machines.

Personally, I use the Anaconda Python distribution in all of my work, [available here](https://store.continuum.io/cshop/anaconda/). I am not selecting them out of favoritism, I am merely documenting my environment. Should you have trouble running any of the code, perhaps knowing this will help you.

Provided Libraries
-----

I've not structured anything nicely yet. For now just look for any .py files in the base directory. As I pull everything together I will turn this into a python library, and probably create a separate git project just for the python code.

There are python files with a name like *xxx*_internal.py. I use these to store functions that are useful for the book, but not of general interest. Often the Python is the point and focus of what I am talking about, but sometimes I just want to display a chart. IPython Notebook does not allow you to collapse the python code, and so it sometimes gets in the way. Some IPython books just incorporate .png files for the image, but I want to ensure that everything is open - if you want to look at the code you can. 

Some chapters introduce functions that are useful for the rest of the book. Those functions are initially defined within the Notebook itself, but the code is also stored in a Python file that is imported if needed in later chapters. I do document when I do this where the function is first defined. But this is still a work in progress.


License
-----

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="http://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Kalman Filters and Random Signals in Python</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python" property="cc:attributionName" rel="cc:attributionURL">Roger Labbe</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python" rel="dct:source">https://github.com/rlabbe/Kalman-Filters-and-Random-Signals-in-Python</a>.


Contact
-----
