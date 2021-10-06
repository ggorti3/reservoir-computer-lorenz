import numpy as np
import matlab.engine
import matlab
import random

def RK4(f, r0, tf, dt):
    """Fourth-order Runge-Kutta integrator.

    :param f: Function to be integrated
    :param r0: Initial conditions
    :param tf: Integration duration
    :param dt: Timestep size
    :returns: time and trajectory vectors

    """
    
    # generate an array of time steps
    ts = np.arange(0, tf, dt)
    
    # create an array to hold system state at each timestep
    traj = np.zeros((ts.shape[0], len(r0)))
    traj[0, :] = np.array(r0)
    
    # calculate system state at each time step, save it in the array
    for i in range(0, ts.shape[0]-1):
        t = ts[i]
        r = traj[i, :]

        k1 = dt * f(r, t)
        k2 = dt * f(r + k1/2, t + dt/2)
        k3 = dt * f(r + k2/2, t + dt/2)
        k4 = dt * f(r + k3, t + dt)
        K = (1.0/6)*(k1 + 2*k2 + 2*k3 + k4)

        traj[i+1, :] = r + K
    
    return (ts, traj)

def generateLorenz(r0, tf, dt, sigma, rho, beta):
    """Integrate a given Lorenz system."""

    # define equations of lorenz system
    def lorenz(r, t):
        x = r[0]; y = r[1]; z = r[2]
        u = sigma * (y - x)
        v = x * (rho - z) - y
        w = x * y - beta * z
        return np.array([u, v, w])

    ts, traj = RK4(lorenz, r0, tf, dt)
    return (ts, traj)

def get_lorenz_data(tf=250, dt=0.02, skip=25, split=0.8):
    _, traj = generateLorenz((1, 1, 1), tf, dt, 10, 28, 8/3)
    
    skip_steps = int(25 / dt)
    traj = traj[skip_steps:]
    
    split_num = int(split * traj.shape[0])
    
    train_data = traj[:split_num]
    val_data = traj[split_num:]
    
    return train_data, val_data

def get_KS_data(num_gridpoints=128, tf=2000, dt=0.25, skip=25, split=0.8, seed=1):
    random.seed(1)
    u0_list = []
    for i in range(num_gridpoints):
        u0_list.append([2 * (random.random() - 0.5)])
    u0 = matlab.double(u0_list)
    eng = matlab.engine.start_matlab()
    traj = eng.evolve_KS(u0, tf, dt, "full")
    skip_steps = int(skip / dt)
    traj = np.array(traj).transpose()[skip_steps:]

    split_num = int(split * traj.shape[0])
    train_data = traj[:split_num]
    val_data = traj[split_num:]

    return train_data, val_data