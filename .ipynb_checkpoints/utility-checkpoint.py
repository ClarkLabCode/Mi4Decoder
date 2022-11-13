# Utility functions and classes for the Mi4 simulation

import numpy as np
import scipy as sp

# Function to create checkerboard stimulus (space 1D, time 1D)
def create_checker(
        contrast=1,
        update_rate=0, # Hz
        resolution=5, # degrees
        x_extent=360, # degrees
        t_extent=8, # seconds
        on_off_t=(1,6), # seconds
        dx=1, # degrees
        dt=0.01 # seconds
    ):
    
    # we assume dx to cleanly divide resolution
    if np.mod(resolution, dx) != 0:
        print('dx should clearnly divide resolution!')
        return 0
    
    # Calculate the numbers of the samples
    n_samples_x = int(np.ceil(x_extent / dx))
    n_samples_t = int(np.ceil(t_extent / dt))
    
    # Create the mesh 
    x_vec = np.linspace(0, x_extent, n_samples_x, endpoint=False)
    t_vec = np.linspace(0, t_extent, n_samples_t, endpoint=False)
    x_mat, t_mat = np.meshgrid(x_vec, t_vec)
    
    # Preapre stimulus related variables
    n_checker = int(np.ceil(x_extent/resolution))
    checker_resize_ratio = int(resolution/dx)
    
    if update_rate>0:
        t_updates = np.arange(on_off_t[0], on_off_t[1], 1.0/update_rate)
        t_updates = np.append(t_updates,(on_off_t[1]))
    else:
        t_updates = np.asarray(on_off_t)
    
    stim = []
    
    n_t_samples_pre = np.sum(t_vec<t_updates[0])
    n_t_samples_post = np.sum(t_vec>t_updates[-1])

    # pre stimulus grey period
    stim.append(np.zeros((n_t_samples_pre,n_samples_x)))
    
    # add checkers
    for i in range(len(t_updates)-1):
        this_checker_lowres = np.random.randint(2, size=n_checker)
        this_checker_highres = np.kron(this_checker_lowres,np.ones((1,checker_resize_ratio)))[:n_samples_x]
        n_t_samples = np.sum(np.logical_and(t_vec>t_updates[i], t_vec<=t_updates[i+1]))
        this_checker_xt = np.tile(this_checker_highres,(n_t_samples,1)).astype('float') * 2 - 1
        stim.append(this_checker_xt)
    
    # post stimulus grey period
    stim.append(np.zeros((n_t_samples_post,n_samples_x)))
    
    # turn into 2darray
    stim = np.vstack(stim)* contrast
    
    return stim