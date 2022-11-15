# Utility functions and classes for the Mi4 simulation

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

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

# Function to simulate photoreceptor responses & implement downsampling
# convolve gaussian over space, downsample

def photoreceptor_response(
        stim,
        stim_dx,
        sample_dx = 5.0,
        fwhm = 5.0
    ):
    
    # calculate indices for downsampling
    n_sample_x = stim.shape[1]
    sample_index = np.arange(0, n_sample_x, sample_dx / stim_dx, dtype=int)
    
    # Create gaussian filter
    # convert fwhm to sigma
    sigma = fwhm / 2.355
    # prepare x vector with 8 sigma wide (maybe overkill but just in case)
    x_vec = np.arange(-np.ceil(4*sigma), np.ceil(4*sigma)+1, stim_dx)
    # calculate filter
    x_filt = np.exp(-0.5 * x_vec**2 / (sigma**2)) / sigma / np.sqrt(2.0 * np.pi)
    # to 2darray
    x_filt = np.reshape(x_filt,(1,len(x_filt)))
    
    # Do the convolution
    resp = sp.signal.convolve2d(stim, x_filt, mode='same', boundary='wrap')[:, sample_index] * stim_dx
    
    return resp

# Function to simulate Mi4 response given photoreceptor responses
# model temporal filter as t*exp(-t/tau)
def mi4_response(
        pr_resp,
        dx,
        dt,
        sigmas = (5,10),
        weights = (1,0.5),
        tau=0.300 # seconds
    ):
    
    # Create temporal filter
    # length is 10 tau
    n_t_sample = int(tau / dt * 10)
    t_vec = np.arange(n_t_sample) * dt
    t_filt = np.exp(-t_vec/tau) * t_vec
    # normalize to unit L1 norm
    t_filt = t_filt / (np.sum(t_filt) * dt)
    # to 2darray
    t_filt = np.reshape(t_filt, (len(t_filt),1))
    
    # Create spatial filter as difference of gaussian (go for 4 sigma both ways)
    n_x_sample_half = np.ceil(np.max(sigmas) / dx * 4)
    x_vec = np.arange(-n_x_sample_half, n_x_sample_half+1) * dx
    # calculate both gaussian
    x_filt_pos = np.exp(-0.5 * x_vec**2 / (sigmas[0]**2)) / sigmas[0] / np.sqrt(2.0 * np.pi)
    x_filt_neg = np.exp(-0.5 * x_vec**2 / (sigmas[1]**2)) / sigmas[1] / np.sqrt(2.0 * np.pi)
    x_filt = x_filt_pos * weights[0] - x_filt_neg * weights[1]
    # normalize for unit L2 norm
    x_filt = x_filt / np.sqrt(np.sum(x_filt**2) * dx)
    # to 2darray
    x_filt = np.reshape(x_filt, (1,len(x_filt)))
    
    # apply filter (w/ clipping over time)
    out = sp.signal.convolve2d(pr_resp, t_filt, mode='full', boundary='fill')[:pr_resp.shape[0],:] * dt
    out = sp.signal.convolve2d(out, x_filt, mode='same', boundary='wrap') * dx
    
    
    return out, (t_filt, t_vec), (x_filt, x_vec)


# Visualiation utility function
def set_image_axis(ax, im, dx, x_extent, dt, t_extent, t_offset):
    
    x_tick_points = np.linspace(0, x_extent, 5) / dx # tick every 45 degrees
    x_tick_num = np.linspace(0, x_extent, 5) - x_extent/2 # centering
    y_tick_points = np.linspace(0, t_extent,t_extent+1) / dt # tick every second
    y_tick_num = np.linspace(0, t_extent,t_extent+1) - t_offset
    
    ax.set_xticks(x_tick_points)
    ax.set_xticklabels(x_tick_num)
    ax.set_yticks(y_tick_points)
    ax.set_yticklabels(y_tick_num)
    ax.set_xlabel('space (deg)')
    ax.set_ylabel('time (s)')
    
    plt.colorbar(im,location='top')