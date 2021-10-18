import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import random_combo, lin_reg, generate_reservoir

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

    def readout(self):
        with torch.no_grad():
            pred = self.linear(self.r_state).flatten()
            if self.padding > 0:
                assisted_vec = torch.cat([pred[pred.shape[0] - self.padding:], pred, pred[:self.padding]])
            else:
                assisted_vec = pred
            
            return pred, assisted_vec

    def advance(self, assisted_vec):
        with torch.no_grad():
            out = self.conv(assisted_vec.unsqueeze(0).unsqueeze(0))
            self.r_state = self.activation(out.squeeze(0).transpose(0, 1))
    
    def activation(self, x):
        x = self.sig(x)
        x[... , self.combo] = x[... , self.combo] ** 2
        return x

    def predict(self, steps):
        pred_list = []
        for i in range(steps):
            pred, assisted_vec = self.readout()
            pred_list.append(pred.detach().cpu().numpy())
            self.advance(assisted_vec)
        predicted = np.stack(pred_list)
        return predicted

class SCRLC(CRLC):
    """
    CRLC with sub_length 1 and conv filters with reflectional symmetry so that the network encodes translational and
    reflective symmetry.
    """
    def __init__(self, n, padding, sigma=0.1):
        super(SCRLC, self).__init__(n=n, sub_length=1, padding=padding, sigma=sigma)
        c_1 = 2 * sigma * (np.random.rand(n, padding) - 0.5)
        c_3 = c_1[:, ::-1].copy()
        c_2 = 2 * sigma * (np.random.rand(n, 1) - 0.5)
        W_in = np.concatenate([c_1, c_2, c_3], axis=1)
        self.conv.weight.data = torch.tensor(W_in, dtype=torch.double).unsqueeze(1)

class CRC(CRLC):
    """
    CRLC with a reservoir.
    """
    def __init__(self, n, sub_length, padding, rho, density, sigma=0.1):
        super(CRC, self).__init__(n=n, sub_length=sub_length, padding=padding, sigma=sigma)
        A = generate_reservoir(n, rho, density)
        self.A = nn.Linear(n, n, bias=False, dtype=torch.double)
        self.A.weight.data = torch.tensor(A, dtype=torch.double)

def train(model, dataloader, last_vec, lr, epochs=100):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    interval = 50
    for e in range(epochs):
        print("Starting epoch {}".format(e))
        cum_loss = 0
        for i, (batch, batch_label) in enumerate(dataloader):
            model.generate_r_states(batch)
            optimizer.zero_grad()
            out = model.training_forward()
            loss = loss_func(out, batch_label)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                cum_loss += loss
            if i % interval == 0 and i != 0:
                print("    Iteration {} avg loss: {}".format(i, cum_loss / interval))
                cum_loss = 0
        print("Epoch {} complete".format(e))

    with torch.no_grad():
        out = model.conv(last_vec.unsqueeze(0))
        model.r_state = model.activation(out.squeeze(0).transpose(0, 1))
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
    from torch_data import KS_to_torch, KS_Dataset
    from torch.utils.data import DataLoader
    from visualization import plot_images, plot_correlations, compare

    padding = 0
    sub_length = 100
    n = 1000
    dt = 0.25

    rho = 1.1
    density = 6 / n

    train_data, val_data = get_KS_data(num_gridpoints=100, tf=1000, dt=dt)
    training_traj, target = KS_to_torch(train_data, padding, sub_length)
    # torch_dset = KS_Dataset(train_data, padding, sub_length)
    # dataloader = DataLoader(torch_dset, collate_fn=torch_dset.collate_fn, batch_size=40, shuffle=True)
    print("data generated.")

    print("instantiating model...")
    #model = CRLC(n, sub_length, padding)
    #model = SCRLC(n, padding)
    model = CRC(n, sub_length, padding, rho, density)
    print("generating training r_states...")
    model.generate_r_states(training_traj)

    print("fitting model...")
    model.train_normal_eq(target)

    #train(model, dataloader, torch_dset.last_vec, lr=1e-5, epochs=10)

    predicted = model.predict(val_data.shape[0])
    plot_images(predicted, val_data, 500)
    
