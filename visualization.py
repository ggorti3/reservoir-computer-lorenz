import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from utils import correlation_distance
from scipy.signal import argrelmin, argrelmax

### functions for plotting results ###

def compare(predicted, actual, t, fontsize = 10):
    """
    Plot a comparison between a predicted trajectory and actual trajectory.
    
    Plots up to 9 dimensions
    """
    dimensions = predicted.shape[1]
    plt.clf()
    plt.ion()
    
    i = 0
    while i < min(dimensions, 9):
        if i == 0:
            var = "x"
        elif i == 1:
            var = "y"
        elif i == 2:
            var = "z"
        else:
            var = ("dimension {}" .format((i + 1)))
        
        plt.subplot(min(dimensions, 9), 1, (i + 1))
        plt.plot(t, actual[:, i])
        plt.plot(t, predicted[:, i])
        plt.ylabel(var, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if i == 0:
            plt.legend(["truth", "prediction"])
            plt.title("Truth vs Predicted Trajectory Comparison")
        i += 1
        
    plt.xlabel("Time", fontsize=fontsize)
    plt.show()
    input("Press enter to exit")
    
def plot_poincare(predicted):
    """
    Displays the poincare plot of the given predicted trajectory
    """
    plt.clf()
    plt.ion()   
    
    zp = predicted[:, 2]
    
    zpmaxes = zp[argrelextrema(zp, np.greater)[0]]
    zpi = zpmaxes[0:(zpmaxes.shape[0] - 1)]
    zpi1 = zpmaxes[1:]
    
    plt.scatter(zpi, zpi1)
    plt.xlabel("z_i")
    plt.ylabel("z_(i + 1)")
    plt.title("Poincare Plot")
    plt.show()
    input("Press enter to exit")

def plot_images(predicted, actual, num_preds):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    vmin, vmax = -3, 3
    ax1.imshow(actual.transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax1.set_title("Truth")
    ax2.imshow(predicted.transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax2.set_title("Prediction")
    ax3.imshow((actual - predicted).transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax3.set_title("Difference")
    plt.show()

def plot_correlations(traj):
    num_gridpoints = traj.shape[1]
    my_range = np.array(range(num_gridpoints))
    c_dists = []
    for i in range(num_gridpoints):
        c_vec = correlation_distance(traj, i)
        c_dists.append(np.mean(c_vec))

    c_dists = np.array(c_dists)

    max_idxs = argrelmax(c_dists[:c_dists.shape[0]//2 + 1])[0]
    min_idxs = argrelmin(c_dists[:c_dists.shape[0]//2 + 1])[0]
    extrema = np.sort(np.concatenate([np.array([1]), c_dists[max_idxs], np.absolute(c_dists[min_idxs])]))[::-1]
    extrema_idxs = np.sort(np.concatenate([np.array([0]), my_range[max_idxs], my_range[min_idxs]]))

    coeffs = np.polyfit(extrema_idxs, np.log(extrema), 1)
    print(coeffs)
    envelope = np.exp(coeffs[0] * my_range)

    fig, ax = plt.subplots(1)
    ax.plot(my_range, c_dists)
    ax.plot(my_range, envelope)
    ax.set_title("correlation at each distance")
    ax.set_xticks(range(0, num_gridpoints + 1, 10))
    plt.show()
