import numpy as np
import pylab as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Arrow, FancyArrow
import book_plots
import numpy as np
from numpy.random import randn, random, uniform, multivariate_normal, seed
from nonlinear_plots import plot_monte_carlo_mean
import scipy

def plot_random_pd():
    def norm(x, x0, sigma):
        return np.exp(-0.5 * (x - x0) ** 2 / sigma ** 2)


    def sigmoid(x, x0, alpha):
        return 1. / (1. + np.exp(- (x - x0) / alpha))

    x = np.linspace(0, 1, 100)
    y2 =  (0.1 * np.sin(norm(x, 0.2, 0.05)) +  0.25 * norm(x, 0.6, 0.05) +
          .5*norm(x, .5, .08) +
           np.sqrt(norm(x, 0.8, 0.06)) +0.1 * (1 - sigmoid(x, 0.45, 0.15)))
    with plt.xkcd():
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



class ParticleFilter(object):

    def __init__(self, N, x_dim, y_dim):
        self.particles = np.empty((N, 3))  # x, y, heading
        self.N = N
        self.x_dim = x_dim
        self.y_dim = y_dim

        # distribute particles randomly with uniform weight
        self.weights = np.empty(N)
        self.weights.fill(1./N)
        self.particles[:, 0] = uniform(0, x_dim, size=N)
        self.particles[:, 1] = uniform(0, y_dim, size=N)
        self.particles[:, 2] = uniform(0, 2*np.pi, size=N)


    def create_particles(self, mean, variance):
        """ create particles with the specified mean and variance"""
        self.particles[:, 0] = mean[0] + randn(self.N) * np.sqrt(variance)
        self.particles[:, 1] = mean[1] + randn(self.N) * np.sqrt(variance)

    def create_particle(self):
        """ create particles uniformly distributed over entire space"""
        return [uniform(0, self.x_dim), uniform(0, self.y_dim), 0, 0]


    '''def assign_speed_by_gaussian(self, speed, var):
        """ move every particle by the specified speed (assuming time=1.)
        with the specified variance, assuming Gaussian distribution. """

        self.particles[:, 2] = np.random.normal(speed, var, self.N)'''

    def control(self, dx):
        self.particles[:, 0] += dx[0]
        self.particles[:, 1] += dx[1]



        self.particles[:, 1] = (self.particles[:, 1] + vy*dt)


    def predict(self, u, std):
        """ move according to control input u with noise std"""

        self.particles[:, 2] += u[0] + randn(self.N) * std[0]
        self.particles[:, 2] %= 2 * np.pi

        d = u[1] + randn(self.N)
        self.particles[:, 0] += np.cos(self.particles[:, 2]) * d
        self.particles[:, 1] += np.sin(self.particles[:, 2]) * d

        self.particles[:, 0:2] += u + randn(self.N, 2) * std


    def weight(self, z, var):
        dist = np.sqrt((self.particles[:, 0] - z[0])**2 +
                       (self.particles[:, 1] - z[1])**2)

        # simplification assumes variance is invariant to world projection
        n = scipy.stats.norm(0, np.sqrt(var))
        prob = n.pdf(dist)

        # particles far from a measurement will give us 0.0 for a probability
        # due to floating point limits. Once we hit zero we can never recover,
        # so add some small nonzero value to all points.
        prob += 1.e-12
        self.weights += prob
        self.weights /= sum(self.weights) # normalize


    def neff(self):
        return 1. / np.sum(np.square(self.weights))


    def resample(self):
        p = np.zeros((self.N, 3))
        w = np.zeros(self.N)

        cumsum = np.cumsum(self.weights)
        for i in range(self.N):
            index = np.searchsorted(cumsum, random())
            p[i] = self.particles[index]
            w[i] = self.weights[index]

        self.particles = p
        self.weights = w / np.sum(w)


    def estimate(self):
        """ returns mean and variance """
        pos = self.particles[:, 0:2]
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu)**2, weights=self.weights, axis=0)

        return mu, var




def plot_pf(pf, xlim=100, ylim=100, weights=True):

    if weights:
        a = plt.subplot(221)
        a.cla()

        plt.xlim(0, ylim)
        #plt.ylim(0, 1)
        a.set_yticklabels('')
        plt.scatter(pf.particles[:, 0], pf.weights, marker='.', s=1, color='k')
        a.set_ylim(bottom=0)

        a = plt.subplot(224)
        a.cla()
        a.set_xticklabels('')
        plt.scatter(pf.weights, pf.particles[:, 1], marker='.', s=1, color='k')
        plt.ylim(0, xlim)
        a.set_xlim(left=0)
        #plt.xlim(0, 1)

        a = plt.subplot(223)
        a.cla()

    else:
        plt.cla()
    plt.scatter(pf.particles[:, 0], pf.particles[:, 1], marker='.', s=1, color='k')
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)


def show_two_pf_plots():
    """ Displays results of PF after 1 and 10 iterations for the book.
    Note the book says this solves the full robot localization problem.
    It doesn't bother simulating landmarks as this is just an illustration.
    """

    seed(1234)
    N = 3000
    pf = ParticleFilter(N, 20, 20)
    z = np.array([20, 20])

    #plot(pf, weights=False)

    for x in range(10):

        z[0] = x+1 + randn()*0.3
        z[1] = x+1 + randn()*0.3

        pf.predict((1,1), (0.2, 0.2))
        pf.weight(z=z, var=.8)
        pf.resample()

        if x == 0:
            plt.subplot(121)
        elif x == 9:
            plt.subplot(122)

        if x == 0 or x == 9:
            mu, var = pf.estimate()
            plot_pf(pf, 20, 20, weights=False)
            if x == 0:
                plt.plot(x+1, x+1, marker='*', color='r', ms=10)
                plt.scatter(mu[0], mu[1], color='g', s=100)
            else:
                plt.scatter(mu[0], mu[1], color='g', s=100, label="PF")
                plt.scatter([x+1], [x+1], marker='*', color='r', s=60, label="True")
                plt.legend(scatterpoints=1)
            plt.tight_layout()


if __name__ == '__main__':
    show_two_pf_plots()
