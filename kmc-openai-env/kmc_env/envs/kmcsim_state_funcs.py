# Writing some helper functions

import sys
import os
import numpy as np
import collections


def make_surface_proj(atom_box):
    #Given an atom box (matrix with 1s where atoms are), determine the surface projection
    
    surface_proj = np.zeros(shape=(atom_box.shape[0], atom_box.shape[1]))
    for i in range(surface_proj.shape[0]):
        for j in range(surface_proj.shape[1]):
            try:
                surface_proj[i,j] = np.max(np.where(atom_box[i,j,:]==1))
            except ValueError:
                surface_proj[i,j] = 0
    return surface_proj

def calc_roughness(surface_projection):
    #Calculate the roughness, given a surface projection
    N = surface_projection.shape[0]
    M = surface_projection.shape[1]
    zbar = np.mean(surface_projection)
    z_sum = 0
    for i in range(N):
        for j in range(M):
            z_sum+=((surface_projection[i,j] - zbar )**2)
    rms_roughness = np.sqrt((1.0/(N*M))*z_sum)
    return rms_roughness

def get_state_reward(sim_model, latt, target_roughness):
    '''Given an input of the simulation model this function returns the state and reward'''
    #To get the final state put atoms into atom box
    arr = sim_model.kmc.get_conf()
    arr_1 = np.array(arr[0])

    print('Printing lattice: ', latt['box'])
    full_atom_box = np.zeros([latt['box'][1],latt['box'][2],latt['box'][3] ])
    try:
        for i,j,k in arr_1:
            full_atom_box[i,j,k]=1
    except IndexError:
            print('Warning: IndexError in kmcSim. Diagnose this fault.')

    surface_proj = make_surface_proj(full_atom_box)
    rms_val = calc_roughness(surface_proj)
    reward = -1*np.sqrt((target_roughness-rms_val)**2) #penalty for straying from desired roughness.
    return surface_proj, reward


# +
def get_incremented_rates(existing_rates, action, dep_rates, current_temp):
    #Given some rates and actions, increment appropriately and return updated rates

    new_rates=[]
    #handle the deposition case first
    for ind in range(2):
        rate = existing_rates[ind]
        rate = max(rate+dep_rates[action[ind]],0.01)
        if rate>=0.30: rate = 0.3
        if rate <=0.010: rate = 0.010
        new_rates.append(rate)
        
    #hendling the temperature
    temp_rates = [-50,0,50] #decrease by 50K, maintain or increase by 50K
    new_temp = current_temp + temp_rates[action[2]]
    
    new_diffusion_rates = get_new_diffusion_rates(new_temp)
    
    new_rates.append(new_diffusion_rates[0])
    new_rates.append(new_diffusion_rates[1])
    new_rates.append(new_diffusion_rates[2])
    
    return new_rates, new_temp

def get_new_diffusion_rates(new_temp):
    # The following define the behavior of the diffusion rates with respect to T
    #Best not to change for now.
    
    T = np.linspace(600,1100,501)
    
    #clip T
    new_temp = min(new_temp, max(T))
    new_temp = max(new_temp, min(T))

    rate_same_offset = 0.20 
    rate_mix_offset = 0.09 
    rate_diff_offset = 0.02 

    rate_same_slope  = 1E-4
    rate_mix_slope = 1.9E-4
    rate_diff_slope  = 2.50E-4
    
    diffusion_rate_same = np.array(rate_same_slope*T + rate_same_offset)[np.where(T==new_temp)[0][0]]
    diffusion_rate_different = np.array(rate_diff_slope*T + rate_diff_offset)[np.where(T==new_temp)[0][0]]
    diffusion_rate_mix = np.array(rate_mix_slope*T + rate_mix_offset)[np.where(T==new_temp)[0][0]]
    
    return [diffusion_rate_same, diffusion_rate_different, diffusion_rate_mix]


# -

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
