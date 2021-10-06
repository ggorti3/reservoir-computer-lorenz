import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import random_combo, lin_reg

class CRLC(nn.Module):
    def __init__(self, n, sub_length, padding, sigma=0.1):
        # we can model the operation of the subsampling RLC using a convolution along space at one point in time
        # filters should be size (sub_length, )
        # number of filters = number of nodes in hidden state
        # stride should be size of prediction window (sub_length - 2 * padding)
        # then, a linear layer will map the output of the conv layer to the next system state
        # we will train this linear layer with gradient descent
        super(CRLC, self).__init__()
        self.combo = random_combo(n, n//2)
        self.n = n
        self.sub_length = sub_length
        self.padding = padding

        W_in = 2 * sigma * (np.random.rand(n, sub_length + 2 * padding) - 0.5)
        self.conv = nn.Conv1d(1, n, sub_length + 2 * padding, stride=sub_length, bias=False).type(torch.double)
        self.conv.weight.data = torch.tensor(W_in, dtype=torch.double).unsqueeze(1)
        self.linear = nn.Linear(n, sub_length, bias=False, dtype=torch.double)
        self.sig = nn.Sigmoid()
    
    def generate_r_states(self, traj):
        """
        generate reservoir states for each state vector in training trajectory

        :params:
        traj (torch.tensor): training trajectory (num_samples, 1, len_state_vector + 2 * padding)
        """
        with torch.no_grad():
            num_samples = traj.shape[0]
            r_states = self.conv(traj) # out is (num_samples, n, len_state_vector // sub_length)
            num_subwindows = r_states.shape[2]
            r_states = r_states.transpose(1, 2).reshape((num_samples * num_subwindows, self.n))
            r_states = self.activation(r_states)
            self.r_state = r_states[r_states.shape[0] - num_subwindows:]
            self.training_states = torch.cat([torch.zeros((num_subwindows, self.n)), r_states[:r_states.shape[0] - num_subwindows]], dim=0)
    
    def training_forward(self):
        return self.linear(self.training_states)
    
    def train_normal_eq(self, target):
        """
        using generated reservoir states, find W_out with normal equations

        :params:
        traj (torch.tensor): training trajectory (num_samples, 1, len_state_vector + 2 * padding)
        """
        U = target.cpu().numpy()
        R = self.training_states.cpu().numpy().transpose()
        W_out = lin_reg(R, U, 0.0001)
        self.linear.weight.data = torch.tensor(W_out, dtype=torch.double)

    def pred_forward(self):
        with torch.no_grad():
            pred = self.linear(self.r_state).flatten()
            if self.padding > 0:
                assisted_vec = torch.cat([pred[pred.shape[0] - self.padding:], pred, pred[:self.padding]])
            else:
                assisted_vec = pred
            
            with torch.no_grad():
                out = self.conv(assisted_vec.unsqueeze(0).unsqueeze(0))
                self.r_state = self.activation(out.squeeze(0).transpose(0, 1))
        return pred
    
    def activation(self, x):
        x = self.sig(x)
        x[... , self.combo] = x[... , self.combo] ** 2
        return x

def KS_to_torch(train_data, padding, sub_length):
    # sub length must divide train_data.shape[1]
    training_traj = torch.tensor(train_data, dtype=torch.double)
    if padding > 0:
        training_traj = torch.cat([training_traj[:, training_traj.shape[1] - padding:], training_traj, training_traj[:, :padding]], dim=1)
    training_traj = training_traj.unsqueeze(1)
    target = torch.tensor(train_data, dtype=torch.double).reshape((-1, sub_length))
    return training_traj, target

def train(model, training_traj, target, lr, iterations=10000):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    model.generate_r_states(training_traj)
    cum_loss = 0
    interval = 10
    for i in range(iterations):
        optimizer.zero_grad()
        out = model.training_forward()
        loss = loss_func(out, target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            cum_loss += loss
        if i % interval == 0 and i != 0:
            print("avg loss: {}".format(cum_loss / interval))
            cum_loss = 0
    return model

def predict(model, steps):
    pred_list = []
    for i in range(steps):
        pred = model.pred_forward()
        pred_list.append(pred.detach().cpu().numpy())
    predicted = np.stack(pred_list)
    return predicted

if __name__ == "__main__":
    from data import get_KS_data, get_lorenz_data
    from visualization import plot_images, plot_correlations, compare

    padding = 6
    sub_length = 5
    n = 5000
    dt = 0.25

    train_data, val_data = get_KS_data(num_gridpoints=100, tf=5000, dt=dt)
    training_traj, target = KS_to_torch(train_data, padding, sub_length)
    print("data generated.")

    print("instantiating model...")
    model = CRLC(n, sub_length, padding)
    print("generating training r_states...")
    model.generate_r_states(training_traj)

    print("fitting model...")
    model.train_normal_eq(target)
    out = model.training_forward()

    predicted = predict(model, val_data.shape[0])
    plot_images(predicted, val_data, 500)

    # padding = 0
    # sub_length = 3
    # n = 400
    # dt = 0.02

    # train_data, val_data = get_lorenz_data()
    # training_traj, target = KS_to_torch(train_data, padding, sub_length)
    # print("data generated. training network...")

    # model = CRLC(n, sub_length, padding, sigma=0.1)
    # model.generate_r_states(training_traj)
    # model.train_normal_eq(target)
    # out = model.training_forward()

    # predicted = predict(model, val_data.shape[0])
    # t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])
    # compare(predicted, val_data, t_grid)
    
