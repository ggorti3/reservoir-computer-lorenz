import numpy as np

from utils import generate_reservoir, scale_res, lin_reg, sigmoid

# This file contains the ReservoirComputer class and necessary helper functions
# The main method at the bottom contains a demo of the code, run this file in the terminal to check it out

class ReservoirComputer():
    def __init__(self, dim_system=3, dim_reservoir=300, sigma=0.1, rho=1.1, density=0.05, augment=False):
        self.dim_reservoir = dim_reservoir
        self.dim_system = dim_system
        self.augment = augment
        self.r_state = np.zeros(dim_reservoir)
        self.A = generate_reservoir(dim_reservoir, rho, density)
        self.W_in = 2 * sigma * (np.random.rand(dim_reservoir, dim_system) - 0.5)
        self.W_out = np.zeros((dim_system, dim_reservoir))
    
    def advance(self, u):
        """
        Generate the next r_state given an input u and the current r_state
        """
        
        # We decided to use the sigmoid function instead of the tanh function. It seems to provide more consistent results
        self.r_state = np.tanh(np.dot(self.A, self.r_state) + np.dot(self.W_in, u))
        
    def readout(self):
        """
        Generate and return the prediction v given the current r_state
        """
        if self.augment:
            r_temp = np.concatenate([self.r_state[:(self.dim_reservoir // 2)], self.r_state[(self.dim_reservoir // 2):]**2])
            v = np.dot(self.W_out, r_temp)
        else:
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
        if self.augment:
            R = np.concatenate([R[:(self.dim_reservoir // 2)], R[(self.dim_reservoir // 2):] ** 2])
        self.W_out = lin_reg(R, traj, 0.0001)
        
    def predict(self, steps):
        """
        Use the network to generate a series of predictions
        
        Parameters
        steps: the number of predictions to make. Can be any positive integer
        
        Returns
        predicted: the predicted trajectory stored as a (steps, dim_system) dimensional numpy array
        """
        predicted = np.zeros((steps, self.dim_system))
        for i in range(steps):
            v = self.readout()
            predicted[i] = v
            self.advance(v)
        return predicted
    

if __name__ == "__main__":
    from data import get_lorenz_data, get_KS_data
    from visualization import compare, plot_poincare, plot_images

    dt = 0.25
    #train_data, val_data = get_lorenz_data(tf=250, dt=dt)
    train_data, val_data = get_KS_data(num_gridpoints=100, tf=20000, dt=dt)

    network = ReservoirComputer(dim_reservoir=9000, dim_system=100, rho=0.4, sigma=0.5, augment=True)
    network.train(train_data)
    predicted = network.predict(val_data.shape[0])
    t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])


    # compare(predicted, val_data, t_grid)
    # plot_poincare(predicted)
    plot_images(predicted, val_data)
