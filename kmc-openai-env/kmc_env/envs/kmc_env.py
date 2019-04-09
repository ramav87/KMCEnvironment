# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 19:15:58 2019

@author: rvv
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from kmcsim.buildtools import make_fcc, write_latt
from kmcsim.sim import KMCModel
from kmcsim.sim import EventTree
from kmcsim.sim import RunSim
import os
import numpy as np
import collections
from kmc_env.envs.kmcsim_state_funcs import make_surface_proj,calc_roughness,get_state_reward,get_incremented_rates

class KmcEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.target_roughness = 0.98
        #print('Current directory is {}'.format(os.getcwd()))
        #print ('Current directory of this file is {}'.format(os.path.dirname(__file__)))
        self.wdir = os.path.join(os.path.dirname(__file__),'data/')
        self.dep_rates = [-0.02, 0, 0.02]
        box = [16, 32, 4]
        latt = make_fcc(box)
        # extend the box in the z-direction to make space for new layers to grow
        latt['box'][3] = 12
        self.latt = latt
        sim = RunSim()
        sim.read(os.path.join(self.wdir, 'kmc.input'))
        sim.init_sim()
        self.sim = sim
        self.state, self.reward = get_state_reward(self.sim, self.latt, self.target_roughness)
    
    def step(self, action, verbose=False):
        #Given simulation model and the action, update the rate and continue running the simulation
        s = self.sim
        
        existing_rates = s.kmc.etree.rates
        
        new_updated_rates = get_incremented_rates(existing_rates, action, self.dep_rates)
        
        s.update_rate(np.array(new_updated_rates), verbose=verbose)
        
        end_flag = s.run_to_next_step(random_seed = np.random.randint(1,99))
        
        #Now get the state and reward
        state, reward = get_state_reward(s, self.latt, self.target_roughness)
        
        #Override the reward here
        if not end_flag: #check the end flag if it is true/false or 1/0.
            reward = -1
        else:
        #add the stuff from the notebook
        
        return state, reward, end_flag
    
    def reset(self):
        sim = RunSim()
        print('Current directory is {}'.format(os.getcwd()))
        sim.read(os.path.join(self.wdir, 'kmc.input'))
        sim.init_sim()
        self.sim = sim
        
        return 1
    
    def render(self,mode='human', close=False):
        s = self.sim
        arr = s.kmc.get_conf()
        arr_1 = np.array(arr[0])
        latt = self.latt
        full_atom_box = np.zeros([latt['box'][1],latt['box'][2],latt['box'][3] ])
        for i,j,k in arr_1:
            full_atom_box[i,j,k]=1
        
        surface_proj = make_surface_proj(full_atom_box)
        rms_val = calc_roughness(surface_proj)
        results = {'atom_box': full_atom_box, 'rms_val': rms_val,
'surface_proj': surface_proj, 'end_flag':end_flag}

        return results
