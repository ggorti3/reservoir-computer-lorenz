import numpy as np
import torch
from torch.utils.data import Dataset

def KS_to_torch(train_data, padding, sub_length):
    # sub length must divide train_data.shape[1]
    training_traj = torch.tensor(train_data, dtype=torch.double)
    if padding > 0:
        training_traj = torch.cat([training_traj[:, training_traj.shape[1] - padding:], training_traj, training_traj[:, :padding]], dim=1)
    training_traj = training_traj.unsqueeze(1)
    target = torch.tensor(train_data, dtype=torch.double).reshape((-1, sub_length))
    return training_traj, target

class KS_Dataset(Dataset):
    def __init__(self, traj, padding, sub_length):
        """
        traj is np array
        """
        self.sub_length = sub_length
        self.padding = padding
        self.training_traj, self.target = KS_to_torch(traj, padding, sub_length)
        self.last_vec = self.training_traj[-1]
        self.num_windows = (self.training_traj.shape[2] - 2 * self.padding) // sub_length
    
    def __len__(self,):
        return self.training_traj.shape[0] - 1
    
    def __getitem__(self, idx):
        return self.training_traj[idx], self.target[self.num_windows * idx:self.num_windows * (idx + 1)]
    
    def collate_fn(self, batch_list):
        x_list = []
        label_list = []
        for x, label in batch_list:
            x_list.append(x)
            label_list.append(label)
        return torch.stack(x_list), torch.cat(label_list, dim=0)

if __name__ == "__main__":
    from data import get_KS_data
    from torch.utils.data import DataLoader

    padding = 6
    sub_length = 10
    dt = 0.25

    train_data, val_data = get_KS_data(num_gridpoints=100, tf=1000, dt=dt)
    training_traj, target = KS_to_torch(train_data, padding, sub_length)
    torch_dset = KS_Dataset(train_data, padding, sub_length)
    dataloader = DataLoader(torch_dset, collate_fn=torch_dset.collate_fn, batch_size=400, shuffle=True)

    batch, label = next(iter(dataloader))
    print(batch.shape)
    print(label.shape)
    print(torch_dset.last_vec.shape)