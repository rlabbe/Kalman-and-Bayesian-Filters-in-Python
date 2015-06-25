import numpy as np
import pylab as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Arrow, FancyArrow
import book_plots
import numpy as np
from numpy.random import multivariate_normal
from nonlinear_plots import plot_monte_carlo_mean

def plot_random_pd():
    def norm(x, x0, sigma):
        return np.exp(-0.5 * (x - x0) ** 2 / sigma ** 2)


    def sigmoid(x, x0, alpha):
        return 1. / (1. + np.exp(- (x - x0) / alpha))

    x = np.linspace(0, 1, 100)
    y2 =  (0.1 * np.sin(norm(x, 0.2, 0.05)) +  0.25 * norm(x, 0.6, 0.05) + 
           np.sqrt(norm(x, 0.8, 0.06)) +0.1 * (1 - sigmoid(x, 0.45, 0.15)))
    plt.xkcd()
    #plt.setp(plt.gca().get_xticklabels(), visible=False)
    #plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.axes(xticks=[], yticks=[], frameon=False)
    plt.plot(x, y2)

def plot_monte_carlo_ukf():

    def f(x,y):
        return x+y, .1*x**2 + y*y
       
    mean = (0, 0)
    p = np.array([[32, 15], [15., 40.]])

    # Compute linearized mean
    mean_fx = f(*mean)

    #generate random points
    xs, ys = multivariate_normal(mean=mean, cov=p, size=3000).T
    fxs, fys = f(xs, ys)
 
    plt.subplot(121)
    plt.gca().grid(b=False)
     
    plt.scatter(xs, ys, marker='.', alpha=.2, color='k')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    plt.subplot(122)
    plt.gca().grid(b=False)

    plt.scatter(fxs, fys, marker='.', alpha=0.2, color='k')
                
    plt.ylim([-10, 200])
    plt.xlim([-100, 100])
    plt.show()