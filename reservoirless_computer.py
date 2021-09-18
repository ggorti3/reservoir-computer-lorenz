import numpy as np
import matplotlib.pyplot as plt
import random

from utils import lin_reg, sigmoid, relu, correlation_distance

def random_combo(n, combo_len):
    assert combo_len < n
    assert combo_len > 0
    my_range = list(range(n))
    combo = []
    for i in range(combo_len):
        randi = random.randrange(len(my_range))
        combo.append(my_range.pop(randi))
    return combo

def random_partition(n, num_cells):
    assert num_cells > 0
    assert num_cells <= n
    my_range = list(range(n))
    cells = [[] for j in range(num_cells)]
    for i in range(n):
        randi = random.randrange(len(my_range))
        cells[i % num_cells].append(my_range.pop(randi))
    return cells


class ReservoirlessComputer():
    def __init__(self, dim_system=3, dim_reservoir=300, sigma=0.1, beta=0.0001, activations=[sigmoid, lambda x : sigmoid(x)**2]):
        self.dim_reservoir = dim_reservoir
        self.dim_system = dim_system
        self.beta = beta
        self.r_state = np.zeros(dim_reservoir)
        self.W_in = 2 * sigma * (np.random.rand(dim_reservoir, dim_system) - 0.5)
        self.W_out = np.zeros((dim_system, dim_reservoir))

        self.activations = activations
        self.num_partitions = len(self.activations)
        self.r_idx_partition = np.array(random_partition(dim_reservoir, self.num_partitions))
    
    def advance(self, u):
        """
        Generate the next r_state given an input u and the current r_state
        """
        
        r_temp = np.dot(self.W_in, u)
        for i in range(self.num_partitions):
            cell = self.r_idx_partition[i]
            activation = self.activations[i]
            r_temp[cell] = activation(r_temp[cell])
        self.r_state = r_temp
        
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
        predicted = np.zeros((steps, self.dim_system))
        for i in range(steps):
            v = self.readout()
            predicted[i] = v
            self.advance(v)
        return predicted

class AssistedReservoirlessComputer(ReservoirlessComputer):
    def __init__(self, dim_system=3, dim_reservoir=300, sigma=0.1, beta=0.0001, activations=[sigmoid, lambda x : sigmoid(x)**2], pad=1):
        super().__init__(dim_system, dim_reservoir, sigma, beta, activations)
        self.W_in = 2 * sigma * (np.random.rand(dim_reservoir, dim_system + 2*pad) - 0.5)
        self.pad = pad
    
    def train(self, assisted_traj):
        # overwrite to implement assisted training
        if assisted_traj.shape[1] != self.dim_system + 2 * self.pad:
            raise ValueError("assisted train data is of wrong dimensions. Given shape: {}. Needed Shape: (n, {})".format(assisted_traj.shape, self.dim_system + 2))
        R = np.zeros((self.dim_reservoir, assisted_traj.shape[0]))
        for i in range(assisted_traj.shape[0]):
            R[:, i] = self.r_state
            x = assisted_traj[i]
            self.advance(x)
        self.W_out = lin_reg(R, assisted_traj[:, self.pad:assisted_traj.shape[1]-self.pad], self.beta)
    
    def train_joint(self, assisted_traj_list):
        cat_assisted_traj = np.concatenate(assisted_traj_list)
        R = np.zeros((self.dim_reservoir, cat_assisted_traj.shape[0]))
        for i, assisted_traj in enumerate(assisted_traj_list):
            self.r_state = np.zeros((self.dim_reservoir, ))
            for j in range(assisted_traj.shape[0]):
                R[:, i*assisted_traj.shape[0] + j] = self.r_state
                x = assisted_traj[j]
                self.advance(x)
        self.W_out = lin_reg(R, cat_assisted_traj[:, self.pad:self.pad + self.dim_system], self.beta)
        num_traj = assisted_traj.shape[0]
        r_state_list = []
        for i in range(len(assisted_traj_list)):
            r_state_list.append(R[:, (i+1)*assisted_traj.shape[0] - 1])
        return r_state_list
    
    def predict(self, steps, assisted_val_data):
        # overwrite to implement assisted prediction
        if assisted_val_data.shape[1] != self.dim_system + 2 * self.pad:
            raise ValueError("assisted val data is of wrong dimensions")
        if steps > assisted_val_data.shape[0]:
            raise ValueError("Too many steps for given val data")
        if steps < 0:
            raise ValueError("Steps cannot be negative")
        predicted = np.zeros((steps, self.dim_system))
        for i in range(steps):
            v = self.readout()
            predicted[i] = v
            assisted_v = np.concatenate([np.atleast_1d(assisted_val_data[i, :self.pad]), v, np.atleast_1d(assisted_val_data[i, self.pad+self.dim_system:])])
            self.advance(assisted_v)
        return predicted
    
def get_subsample(traj, num_gridpoints, start_gridpoint=0):
    if num_gridpoints > traj.shape[1]:
        raise ValueError("Number of grid points to subsample is too large")
    if start_gridpoint + num_gridpoints > traj.shape[1]:
        raise ValueError("Requested subsample is out of bounds")
    if start_gridpoint >= 0:
        assisted_subsample = traj[:, start_gridpoint:(start_gridpoint+num_gridpoints)]
    else:
        # implement wraparound
        assisted_subsample = np.concatenate([traj[:, start_gridpoint:], traj[:, 0:(start_gridpoint+num_gridpoints)]], axis=1)
    return assisted_subsample

def overlap_train(networks, traj_list, steps, val_data):
    pad = networks[0].pad
    for i, net in enumerate(networks):
        net.train(traj_list[i])
    
    pred_traj_list = []
    for i in range(steps):
        pred_vecs = []
        for net in networks:
            pred_vecs.append(net.readout())
        
        assist_vecs = [np.array([]) for j in range(len(networks))]
        # create the assist vecs
        for j in range(len(traj_list)):
            assist_vecs[j] = np.concatenate([pred_vecs[(j - 1) % len(networks)][-1*pad:], pred_vecs[j], pred_vecs[(j + 1) % len(networks)][:pad]])
        
        for j, net in enumerate(networks):
            net.advance(assist_vecs[j])

        pred_traj_list.append(np.concatenate(pred_vecs))

    return np.stack(pred_traj_list)

def overlap_train_joint(net, traj_list, steps, val_data):
    pad = net.pad
    r_state_list = net.train_joint(traj_list)
    
    pred_traj_list = []
    for i in range(steps):
        pred_vecs = []
        for j in range(len(traj_list)):
            net.r_state = r_state_list[j]
            pred_vecs.append(net.readout())
        
        assist_vecs = [np.array([]) for j in range(len(traj_list))]
        # create the assist vecs
        for j in range(len(traj_list)):
            assist_vecs[j] = np.concatenate([pred_vecs[(j - 1) % len(traj_list)][-1*pad:], pred_vecs[j], pred_vecs[(j + 1) % len(traj_list)][:pad]])
        
        for j in range(len(traj_list)):
            net.r_state = r_state_list[j]
            net.advance(assist_vecs[j])
            r_state_list[j] = net.r_state

        pred_traj_list.append(np.concatenate(pred_vecs))

    return np.stack(pred_traj_list)


if __name__ == "__main__":
    from data import get_lorenz_data, get_KS_data
    from visualization import compare, plot_poincare, plot_images, plot_correlations

    dt = 0.25
    #train_data, val_data = get_lorenz_data(tf=250, dt=dt)
    train_data, val_data = get_KS_data(num_gridpoints=100, tf=5000, dt=dt)
    # plot_correlations(train_data)
    # exit()
    # train_subsample = get_subsample(train_data, 40, start_gridpoint=-30)
    # val_subsample = get_subsample(val_data, 40, start_gridpoint=-30)
    train_subsample_0 = get_subsample(train_data, 40, start_gridpoint=-10)
    train_subsample_1 = get_subsample(train_data, 40, start_gridpoint=10)
    train_subsample_2 = get_subsample(train_data, 40, start_gridpoint=30)
    train_subsample_3 = get_subsample(train_data, 40, start_gridpoint=50)
    train_subsample_4 = get_subsample(train_data, 40, start_gridpoint=-30)
    train_traj_list = [train_subsample_0, train_subsample_1, train_subsample_2, train_subsample_3, train_subsample_4]
    print("data generated. training network...")

    activations = [
        sigmoid,
        lambda x : sigmoid(x) ** 2,
        # lambda x : sigmoid(x) ** 3,
        # lambda x : sigmoid(x) ** 4,
        # lambda x : sigmoid(x) ** 5,
        # lambda x : sigmoid(x) ** 6,
        # relu,
    ]
    # network = ReservoirlessComputer(dim_reservoir=22000, dim_system=100, activations=activations)
    # network.train(train_data)
    # predicted = network.predict(val_data.shape[0])
    #t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])

    # network = ReservoirlessComputer(dim_reservoir=5000, dim_system=20, activations=activations)
    # network.train(train_subsample)
    # predicted = network.predict(val_subsample.shape[0])
    # plot_images(predicted, val_subsample)

    # pad = 10
    # network = AssistedReservoirlessComputer(dim_reservoir=7500, dim_system=20, activations=activations, pad=pad)
    # network.train(train_subsample)
    # predicted = network.predict(val_subsample.shape[0], val_subsample)
    # plot_images(predicted, val_subsample[:, pad:val_subsample.shape[1]-pad], 500)

    pad = 10
    net = AssistedReservoirlessComputer(dim_reservoir=8000, dim_system=20, activations=activations, pad=pad)
    # net_0 = AssistedReservoirlessComputer(dim_reservoir=4400, dim_system=20, activations=activations, pad=pad)
    # net_1 = AssistedReservoirlessComputer(dim_reservoir=4400, dim_system=20, activations=activations, pad=pad)
    # net_2 = AssistedReservoirlessComputer(dim_reservoir=4400, dim_system=20, activations=activations, pad=pad)
    # net_3 = AssistedReservoirlessComputer(dim_reservoir=4400, dim_system=20, activations=activations, pad=pad)
    # net_4 = AssistedReservoirlessComputer(dim_reservoir=4400, dim_system=20, activations=activations, pad=pad)
    # networks = [net_0, net_1, net_2, net_3, net_4]

    #predicted = overlap_train(networks, train_traj_list, val_data.shape[0], val_data)
    predicted = overlap_train_joint(net, train_traj_list, val_data.shape[0], val_data)
    plot_images(predicted, val_data, 500)


