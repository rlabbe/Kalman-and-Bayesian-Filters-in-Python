#[Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)


Introductory text for Kalman and Bayesian filters. All code is written in Python, and the book itself is written in IPython Notebook (now known as Jupyter) so that you can run and modify the code in the book in place, seeing the results inside the book. What better way to learn?


**"Kalman and Bayesian Filters in Python" looks amazing! ... your book is just what I needed** - Allen Downey, Professor and O'Reilly author of several math and programming textbooks, via twitter.


![alt tag](https://raw.githubusercontent.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/animations/05_dog_track.gif)

What are Kalman and Bayesian Filters?
-----

We measure things in the world with sensors. For example, thermometers measure temperature, anemometers measure wind speed, scales measure weights. We measure distances with things like sonar, lasers, and calipers. 

Unfortunately, no sensor is perfect. For example, a thermometer might only be accurate to within 2 degrees. A laser rangefinder might be accurate to a meter. A bathroom scale might be accurate to 1 kilogram, whereas a kitchen scale might be accurate to a gram or less. 

Furthermore, a sensor might not present all of the information we would like. Suppose we want to know the map coordinates of a structure that we are surveying. A laser rangefinder tells us the distance to an object, but it does not tell us where in the world that object is. The measurement is *relative* to the position of the rangefinder. If the rangefinder does not provide an angle (bearing) to the measured object we can only state that the object lies somewhere within a donut shaped circle around the rangefinder. If we do know the location of the rangefinder and the angle then we can compute the coordinates of the structure. But how do we know the rangefinder's position? With sensors. How do we measure the angle? With sensors. And sensors are inaccurate, so both of those measurements have a degree of error in them.

Is there a way to reduce the amount of error in the measurements? You may have some ideas already. What if you were to take ten measurements and average them? For many kinds of sensor errors this does improve our estimate. You might imagine 

Motivation
-----

This is a book for programmers that have a need or interest in Kalman filtering. The motivation for this book came out of my desire for a gentle introduction to Kalman filtering. I'm a software engineer that spent almost two decades in the avionics field, and so I have always been 'bumping elbows' with the Kalman filter, but never implemented one myself. As I moved into solving tracking problems with computer vision the need became urgent. There are classic textbooks in the field, such as Grewal and Andrew's excellent *Kalman Filtering*. But sitting down and trying to read many of these books is a dismal experience if you do not have the required background. Typically the first few chapters fly through several years of undergraduate math, blithely referring you to textbooks on topics such as Itō calculus, and present an entire semester's worth of statistics in a few brief paragraphs. They are good texts for an upper undergraduate course, and an invaluable reference to researchers and professionals, but the going is truly difficult for the more casual reader. Symbology is introduced without explanation, different texts use different terms and variables for the same concept, and the books are almost devoid of examples or worked problems. I often found myself able to parse the words and comprehend the mathematics of a definition, but had no idea as to what real world phenomena they describe. "But what does that *mean?*" was my repeated thought.

However, as I began to finally understand the Kalman filter I realized the underlying concepts are quite straightforward. A few simple probability rules, some intuition about how we integrate disparate knowledge to explain events in our everyday life and the core concepts of the Kalman filter are accessible. Kalman filters have a reputation for difficulty, but shorn of much of the formal terminology the beauty of the subject and of their math became clear to me, and I fell in love with the topic. 

As I began to understand the math and theory more difficulties present themselves. A book or paper's author makes some statement of fact and presents a graph as proof.  Unfortunately, why the statement is true is not clear to me, nor is the method for making that plot obvious. Or maybe I wonder "is this true if R=0?"  Or the author provides pseudocode at such a high level that the implementation is not obvious. Some books offer Matlab code, but I do not have a license to that expensive package. Finally, many books end each chapter with many useful exercises. Exercises which you need to understand if you want to implement Kalman filters for yourself, but exercises with no answers. If you are using the book in a classroom, perhaps this is okay, but it is terrible for the independent reader. I loathe that an author withholds information from me, presumably to avoid 'cheating' by the student in the classroom.

From my point of view none of this necessary. Certainly if you are designing a Kalman filter for a aircraft or missile you must thoroughly master of all of the mathematics and topics in a typical Kalman filter textbook. I just want to track an image on a screen, or write some code for an Arduino project. I want to know how the plots in the book are made, and chose different parameters than the author chose. I want to run simulations. I want to inject more noise in the signal and see how a filter performs. There are thousands of opportunities for using Kalman filters in everyday code, and yet this fairly straightforward topic is the provenance of rocket scientists and academics.

I wrote this book to address all of those needs. This is not the book for you if you program navigation computers for Boeing or design radars for Raytheon. Go get an advanced degree at Georgia Tech, UW, or the like, because you'll need it. This book is for the hobbiest, the curious, and the working engineer that needs to filter or smooth data. 

This book is interactive. While you can read it online as static content, I urge you to use it as intended. It is written using Jupyter Notebook, which allows me to combine text, Python, and Python output in one place. Every plot, every piece of data in this book is generated from Python that is available to you right inside the notebook. Want to double the value of a parameter? Click on the Python cell, change the parameter's value, and click 'Run'. A new plot or printed output will appear in the book. 

This book has exercises, but it also has the answers. I trust you. If you just need an answer, go ahead and read the answer. If you want to internalize this knowledge, try to implement the exercise before you read the answer. 

This book has supporting libraries for computing statistics, plotting various things related to filters, and for the various filters that we cover. This does require a strong caveat; most of the code is written for didactic purposes. It is rare that I chose the most efficient solution (which often obscures the intent of the code), and in the first parts of the book I did not concern myself with numerical stability. This is important to understand - Kalman filters in aircraft are carefully designed and implemented to be numerically stable; the naive implementation is not stable in many cases. If you are serious about Kalman filters this book will not be the last book you need. My intention is to introduce you to the concepts and mathematics, and to get you to the point where the textbooks are approachable.

Finally, this book is free. The cost for the books required to learn Kalman filtering is somewhat prohibitive even for a Silicon Valley engineer like myself; I cannot believe the are within the reach of someone in a depressed economy, or a financially struggling student. I have gained so much from free software like Python, and free books like those from Allen B. Downey [here](http://www.greenteapress.com/). It's time to repay that. So, the book is free, it is hosted on free servers, and it uses only free and open software such as IPython and mathjax to create the book. 

In Development
--------------
This book is still very much in development. Chapters are incomplete, code and text could be incorrect. I strive for correctness during development, but I fail. Raise an issue on GitHub, or email me, if you find a problem and I will try to fix it as soon as possible.

## Reading Online

The book is written as a collection of Jupyter Notebooks, an interactive, browser based system that allows you to combine text, Python, and math into your browser. There are multiple ways to read these online, listed below.

### GitHub

GitHub is able to render the notebooks directly. The quickest way to view a notebook is to just click on them above. Chapters names are prefaced with numbers to indicate their order 01_gh_filter.ipynb, and so on. Rendering may be a bit slow or imperfect. Notebooks are rendered statically - you can read them, but not modify or run the code.

### binder

I am experimentally trying a new service, binder. binder serves interactive notebooks online, so you can run the code and change the code within your browser without downloading the book or installing Jupyter. I have not experimented with it much. Please raise an issue on my GitHub page if anything fails. I'm not officially supporting this as it is in beta, but I'm quite excited about the possibilities.

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/rlabbe/Kalman-and-Bayesian-Filters-in-Python)


### nbviewer

The website http://nbviewer.org provides an Jupyter Notebook server that renders notebooks stored at github (or elsewhere). The rendering is done in real time when you load the book. You may use [*this nbviewer link*](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb) to access my book via nbviewer. If you read my book today, and then I make a change tomorrow, when you go back tomorrow you will see that change. Notebooks are rendered statically - you can read them, but not modify or run the code.

### SageMathCloud

The Preface contains instructions on how to host the book online, for free, on SageMath's cloud server (cloud.sagemath.com) that allows you to run and alter the code insde the Notebooks. It takes 5 minutes to set up. We are working to make this easier to use. 


Issues or Questions
------

If you have comments, you can write an issue at GitHub so that everyone can read it along with my response. Please don't view it as a way to report bugs only. Alternatively I've created a gitter room for more informal discussion. [![Join the chat at https://gitter.im/rlabbe/Kalman-and-Bayesian-Filters-in-Python](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/rlabbe/Kalman-and-Bayesian-Filters-in-Python?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


The preface available from the link above has all the information in this README and more, so feel free to follow the link now.

PDF Version
-----
I used to keep a PDF version of the book in this repository. However, git does not do well with binary blobs like that, and the repository was getting too big. I deleted it and it's history from git. I am now providing it 
on my google drive [here](https://drive.google.com/file/d/0By_SW19c1BfhSVFzNHc0SjduNzg/view?usp=sharing)

The PDF will usually lag behind what is in github as I don't update it for every minor check in.

As an experiment I have generated a version for 6x9 paper size. I find it a bit easier to read from a PDF viewer. I may or may not continue to support this format. At the moment the section and chapter names don't fit the top of the page. I know how to fix it, but that is not a top priority for me at the moment.

https://drive.google.com/file/d/0By_SW19c1BfhTHRXWFJ1RUtvaDQ/view?usp=sharing


## Downloading the book


However, this book is intended to be interactive and I recommend using it in that form. It's a little more to set up, but worth it. If you install IPython and some supporting libraries  on your computer and then clone this book you will be able to run all of the code in the book yourself. You can perform experiments, see how filters react to different data, see how different filters react to the same data, and so on. I find this sort of immediate feedback both vital and invigorating. You do not have to wonder "what happens if". Try it and see!

The github pages for this project are at https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python You can clone it to your hard drive with the command 

    git clone https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python.git
   
   

This will create a directory named Kalman-and-Bayesian-Filters-in-Python.

Follow the instructions in **Installation and Software Requirements** below to install all the supporting sofware require. Then, navigate to the directory, and run IPython notebook with the command 

    ipython notebook

This will open a browser window showing the contents of the base directory. The book is organized into chapters. To read Chapter 2, click on the link for chapter 2. This will cause the browser to open that subdirectory. In each subdirectory there will be one or more IPython Notebooks (all notebooks have a .ipynb file extension). The chapter contents are in the notebook with the same name as the chapter name. There are sometimes supporting notebooks for doing things like generating animations that are displayed in the chapter. These are not intended to be read by the end user, but of course if you are curious as to how an animation is made go ahead and take a look.

This is admittedly a somewhat cumbersome interface to a book; I am following in the footsteps of several other projects that are somewhat repurposing Jupyter Notebook to generate entire books. I feel the slight annoyances have a huge payoff - instead of having to download a separate code base and run it in an IDE while you try to read a book, all of the code and text is in one place. If you want to alter the code, you may do so and immediately see the effects of your change. If you find a bug, you can make a fix, and push it back to my repository so that everyone in the world benefits. And, of course, you will never encounter a problem I face all the time with traditional books - the book and the code are out of sync with each other, and you are left scratching your head as to which source to trust.

**Breaking change: I have upgraded to IPython 3.0. This release alters the notebook format (.ipynb) files. If you are running an older version you will likely get the unhelpful error message "Bad request" when you try to open the notebook. Note that this is the version number for _IPython_, which provides the IPython Notebook software, and not the Python version. I.e. you can run these notebooks with Python 2.7, so long as you have IPython 3.0 installed. IPython 3.0 was released on Febuary 27, 2015, so if your install is later than that you will have to update IPython. IPython 2.4 can read the files, but not write them. I apologize if you are using an earlier version, but this is an unavoidable change and I'd rather change now instead of later. This will not affect you if you are reading online, only if you are running the notebooks on your local computer. Please note that this has nothing to do with Python 3 - you can run Python 2.7 in IPython 3.**


Companion Software
-----

[![Latest Version](http://img.shields.io/pypi/v/filterpy.svg)](http://pypi.python.org/pypi/filterpy)

All of the filters used in this book as well as others not in this book are implemented in my Python library FilterPy, available [here](https://github.com/rlabbe/filterpy). You do not need to download or install this to read the book, but you will likely want to use this library to write your own filters. It includes Kalman filters, Fading Memory filters, H infinity filters, Extended and Unscented filters, least square filters, and many more.  It also includes helper routines that simplify the designing the matrices used by some of the filters, and other code such as Kalman based smoothers.

The easiest way to install is to run pip from the command line:

    $ pip install filterpy
    

Installation and Software Requirements
-----

**author's note**. *The book is still being written, and so I am not focusing on issues like supporting multiple versions of Python. I am staying more or less on the bleeding edge of Python 3 for the time being. If you follow my suggestion of installing Anaconda the versioning problems will be taken care of for you, and you will not alter or affect any existing installation of Python on your machine. I am aware that telling somebody to install a specific packaging system is not a long term solution, but I can either focus on endless regression testing for every minor code change, or work on delivering the book, and then doing one sweep through it to maximize compatibility. I opt for the latter. In the meantime I welcome bug reports if the book does not work on your platform.*

If you want to run the notebook on your computer, which is what I recommend, then you will have to have IPython installed. I do not cover how to do that in this book; requirements change based on what other Python installations you may have, whether you use a third party package like Anaconda Python, what operating system you are using, and so on. 

To use all features you will have to have IPython 3.0 or later installed, which is released and stable as of April 2014. Most of the book does not require that recent of a version, but I do make use of the interactive plotting widgets introduced in this release. A few cells will not run if you have an older version installed. This is merely a minor annoyance. If you have IPython 4.0 or later installed then you will need to install the notebook separately. As of version 4.0 the notebook was split into a standalone project called Jupyter. If your distribution does not include Jupyter issuing *pip install jupyter* from the command line should install it. As of August 2015 the instructions and documentation for Jupyter is sparse; if you have customized your IPython environment you may have some searching to do to set up your environment for 4.0. I am not prepared to support your 4.0 install as I am focusing on writing the book, though of course if you encounter a bug either I or the Jupyter team would love to hear about it (depending on whose code has the bug, of course!)  

You will need Python 2.7 or later installed. Almost all of my work is done in Python 3.4, but I periodically test on 2.7. I do not promise any specific check in will work in 2.7 however. I do use Python's "from \_\_future\_\_ import ..." statement to help with compatibility. For example, all prints need to use parenthesis. If you try to add, say, "print 3.14" into the book your script will fail; you must write "print (3.4)" as in Python 3.X.

You will need a recent version of NumPy, SciPy, SymPy, and Matplotlib installed. I don't really know what the minimal version might be. I have numpy 1.71, SciPy 0.13.0, and Matplotlib 1.4.0 installed on my machines.

Personally, I use the Anaconda Python distribution in all of my work, [available here](https://store.continuum.io/cshop/anaconda/). I am not selecting them out of favoritism, I am merely documenting my environment. Should you have trouble running any of the code, perhaps knowing this will help you. Especially if you are in Windows, installing the entire stack above can be difficult if you do not use a distribution like Anaconda.

Finally, you will need to install FilterPy, described in the next section.

Installation of all of these packages is described in the Installation appendix, which you can read online [here](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/Appendix_A_Installation.ipynb).


Provided Libraries and Modules
-----

I am writing an open source Bayesian filtering Python library called **FilterPy**. I have made the project available on PyPi, the Python Package Index.  To install from PyPi, at the command line issue the command

    pip install filterpy

If you do not have pip, you may follow the instructions here: https://pip.pypa.io/en/latest/installing.html.

I recommend using pip for the install, as I try hard (sometimes I fail) to make good, safe releases to it. Whereas on github (where the project resides) I'll check in any tiny change at any time; there is no guarantee that the latest on github is a version you really want to have. 

FilterPy is  hosted github at (https://github.com/rlabbe/filterpy).  If you want the bleading edge release you will want to grab a copy from github, and follow your Python installation's instructions for adding it to the Python search path. However, it is quite possible that it will not be compatible with the notebooks in this project. I try to make sure the notebooks are in sync with the PyPi hosted version.


Code that is specific to the book is stored with the book in the subdirectory ./*code*. This code is in a state of flux; I do not wish to document it here yet. I do mention in the book when I use code from this directory, so it should not be a mystery.

In the *code* subdirectory there are Python files with a name like *xxx*_internal.py. I use these to store functions that are useful for a specific chapter. This allows me to hide away Python code that is not particularly interesting to read - I may be generating a plot or chart, and I want you to focus on the contents of the chart, not the mechanics of how I generate that chart with Python. If you are curious as to the mechanics of that, just go and browse the source.

Some chapters introduce functions that are useful for the rest of the book. Those functions are initially defined within the Notebook itself, but the code is also stored in a Python file that is imported if needed in later chapters. I do document when I do this where the function is first defined, but this is still a work in progress. I try to avoid this because then I always face the issue of code in the directory becoming out of sync with the code in the book. However, IPython Notebook does not give us a way to refer to code cells in other notebooks, so this is the only mechanism I know of to share functionality across notebooks.

There is an undocumented directory called **experiments**. This is where I write and test code prior to putting it in the book. There is some interesting stuff in there, and feel free to look at it. As the book evolves I plan to create examples and projects, and a lot of this material will end up there. Small experiments will eventually just be deleted. If you are just interested in reading the book you can safely ignore this directory. 


The directory **code** contains a css file containing the style guide for the book. The default look and feel of IPython Notebook is rather plain. Work is being done on this. I have followed the examples set by books such as [Probabilistic Programming and Bayesian Methods for Hackers](http://nbviewer.ipython.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Chapter1.ipynb). I have also been very influenced by Professor Lorena Barba's fantastic work, [available here](https://github.com/barbagroup/CFDPython). I owe all of my look and feel to the work of these projects. 

License
-----
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Kalman and Bayesian Filters in Python</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python" property="cc:attributionName" rel="cc:attributionURL">Roger R. Labbe</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

All software in this book, software that supports this book (such as in the the code directory) or used in the generation of the book (in the pdf directory) that is contained in this repository is licensed under the following MIT license:

The MIT License (MIT)

Copyright (c) 2015 Roger R. Labbe Jr

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.TION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Contact
-----

rlabbejr at gmail.com

