import numpy as np
import matplotlib.pyplot as plt

from utils import lin_reg, sigmoid

class ReservoirlessComputer():
    def __init__(self, dim_system=3, dim_reservoir=300, sigma=0.1, beta=0.0001):
        self.dim_reservoir = dim_reservoir
        self.dim_system = dim_system
        self.beta = beta
        self.r_state = np.zeros(dim_reservoir)
        self.W_in = 2 * sigma * (np.random.rand(dim_reservoir, 3) - 0.5)
        self.W_out = np.zeros((3, dim_reservoir))
    
    def advance(self, u):
        """
        Generate the next r_state given an input u and the current r_state
        """
        
        # We decided to use the sigmoid function instead of the tanh function. It seems to provide more consistent results
        self.r_state = sigmoid(np.dot(self.W_in, u))
        
    def readout(self):
        """
        Generate and return the prediction v given the current r_state
        """
        v = np.dot(self.W_out, self.r_state)
        return v
    
    def train(self, traj):
        """
        Optimize W_out so that the network can accurately model the given trajectory.
        
        Parameters
        traj: The training trajectory stored as a (n, dim_system) dimensional numpy array, where n is the number of timesteps in the trajectory
        """
        R = np.zeros((self.dim_reservoir, traj.shape[0]))
        for i in range(traj.shape[0]):
            R[:, i] = self.r_state
            x = traj[i]
            self.advance(x)
        self.W_out = lin_reg(R, traj, self.beta)
        
    def predict(self, steps):
        """
        Use the network to generate a series of predictions
        
        Parameters
        steps: the number of predictions to make. Can be any positive integer
        
        Returns
        predicted: the predicted trajectory stored as a (steps, dim_system) dimensional numpy array
        """
        predicted = np.zeros((steps, 3))
        for i in range(steps):
            v = self.readout()
            predicted[i] = v
            self.advance(v)
        return predicted

if __name__ == "__main__":
    from data import get_lorenz_data
    from visualization import compare, plot_poincare

    dt = 0.02
    train_data, val_data = get_lorenz_data(tf=250,dt=dt)

    network = ReservoirlessComputer(dim_reservoir=300)
    network.train(train_data)
    predicted = network.predict(val_data.shape[0])
    t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])
    compare(predicted, val_data, t_grid)
    plot_poincare(predicted)