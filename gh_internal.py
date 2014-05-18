import numpy as np
import pylab as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Arrow, FancyArrow

def create_predict_update_chart(box_bg = '#CCCCCC',
                arrow1 = '#88CCFF',
                arrow2 = '#88FF88'):
    plt.figure(figsize=(6,6), facecolor='w')
    ax = plt.axes((0, 0, 1, 1),
                  xticks=[], yticks=[], frameon=False)
    #ax.set_xlim(0, 10)
    #ax.set_ylim(0, 10)


    pc = Circle((4,5), 0.5, fc=box_bg)
    uc = Circle((6,5), 0.5, fc=box_bg)
    ax.add_patch (pc)
    ax.add_patch (uc)


    plt.text(4,5, "Predict\nStep",ha='center', va='center', fontsize=14)
    plt.text(6,5, "Update\nStep",ha='center', va='center', fontsize=14)

    #btm
    ax.annotate('',
                xy=(4.1, 4.5),  xycoords='data',
                xytext=(6, 4.5), textcoords='data',
                size=20,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none",
                                patchB=pc,
                                patchA=uc,
                                connectionstyle="arc3,rad=-0.5"))
    #top
    ax.annotate('',
                xy=(6, 5.5),  xycoords='data',
                xytext=(4.1, 5.5), textcoords='data',
                size=20,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none",
                                patchB=uc,
                                patchA=pc,
                                connectionstyle="arc3,rad=-0.5"))


    ax.annotate('Measurement ($\mathbf{z_k}$)',
                xy=(6.3, 5.4),  xycoords='data',
                xytext=(6,6), textcoords='data',
                size=18,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    ax.annotate('',
                xy=(4.0, 3.5),  xycoords='data',
                xytext=(4.0,4.5), textcoords='data',
                size=18,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    ax.annotate('Initial\nConditions ($\mathbf{x_0}$)',
                xy=(4.0, 5.5),  xycoords='data',
                xytext=(2.5,6.5), textcoords='data',
                size=18,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    plt.text (4,3.4,'State Estimate ($\mathbf{\hat{x}_k}$)',
              ha='center', va='center', fontsize=18)
    plt.axis('equal')
    #plt.axis([0,8,0,8])
    plt.show()


def plot_estimate_chart_1():
    ax = plt.axes()
    ax.annotate('', xy=[1,159], xytext=[0,158],
                arrowprops=dict(arrowstyle='->', ec='r',shrinkA=6, lw=3,shrinkB=5))
    plt.scatter ([0], [158], c='b')
    plt.scatter ([1], [159], c='r')
    plt.show()


def plot_estimate_chart_2():
    ax = plt.axes()
    ax.annotate('', xy=[1,159], xytext=[0,158],
                arrowprops=dict(arrowstyle='->',
                                ec='r', lw=3, shrinkA=6, shrinkB=5))
    plt.scatter ([0], [158.0], c='k',s=128)
    plt.scatter ([1], [164.2], c='b',s=128)
    plt.scatter ([1], [159], c='r', s=128)
    plt.text (1.0, 158.8, "prediction ($\hat{x}_{t})$", ha='center',va='top',fontsize=18,color='red')
    plt.text (1.0, 164.4, "measurement ($z$)",ha='center',va='bottom',fontsize=18,color='blue')
    plt.text (0, 157.8, "estimate ($\hat{x}_{t-1}$)", ha='center', va='top',fontsize=18)
    plt.show()

def plot_estimate_chart_3():
    ax = plt.axes()
    ax.annotate('', xy=[1,159], xytext=[0,158],
                arrowprops=dict(arrowstyle='->',
                                ec='r', lw=3, shrinkA=6, shrinkB=5))

    ax.annotate('', xy=[1,159], xytext=[1,164.2],
                arrowprops=dict(arrowstyle='-',
                                ec='k', lw=1, shrinkA=8, shrinkB=8))

    est_y = ((164.2-158)*.8 + 158)
    plt.scatter ([0,1], [158.0,est_y], c='k',s=128)
    plt.scatter ([1], [164.2], c='b',s=128)
    plt.scatter ([1], [159], c='r', s=128)
    plt.text (1.0, 158.8, "prediction ($\hat{x}_{t})$", ha='center',va='top',fontsize=18,color='red')
    plt.text (1.0, 164.4, "measurement ($z$)",ha='center',va='bottom',fontsize=18,color='blue')
    plt.text (0, 157.8, "estimate ($\hat{x}_{t-1}$)", ha='center', va='top',fontsize=18)
    plt.text (0.95, est_y, "new estimate ($\hat{x}_{t}$)", ha='right', va='center',fontsize=18)
    plt.show()


if __name__ == '__main__':
    create_predict_update_chart()