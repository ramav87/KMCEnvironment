# -*- coding: utf-8 -*-
"""

Created on Fri Apr  5 19:15:58 2019

@author: rvv

#TODO: Rama - write a method of class to change and rewrite kmc.input and ni.xyz
#TODO: Rama - check why the state output is given twice after end flag
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
from kmc_env.envs.kmcsim_state_funcs import make_surface_proj,calc_roughness,get_state_reward,get_incremented_rates,gaussian

class KmcEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self,box = [16, 32, 4],box_extension=12,target_roughness=0.98,
    reward_type='box',reward_multiplier=1000,reward_tolerance=1,
    rates_spread=0.01,rates_adjustment=1,
    folder_with_params='None'):
        self.target_roughness = target_roughness
        self.end_flag=0
        #print('Current directory is {}'.format(os.getcwd()))
        #print ('Current directory of this file is {}'.format(os.path.dirname(__file__)))
        if folder_with_params=='None':
            self.wdir = os.path.join(os.path.dirname(__file__),'data/')
        else:
            self.wdir=folder_with_params
        self.reward_type=reward_type
        self.reward_multiplier=reward_multiplier
        self.dep_rates = [-0.02, 0, 0.02]*rates_adjustment
        self.reward_tolerance = reward_tolerance
        self.rates_spread=rates_spread
    
        latt = make_fcc(box)
        # extend the box in the z-direction to make space for new layers to grow
        latt['box'][3] = box_extension

        # self.dep_rates = [-0.02, 0, 0.02]
        # box = [16, 32, 4]
        # latt = make_fcc(box)
        # # extend the box in the z-direction to make space for new layers to grow
        # latt['box'][3] = 12
        self.latt = latt
        sim = RunSim()
        sim.read(os.path.join(self.wdir, 'kmc.input'))
        sim.init_sim()
        #sim.t_max = t_max
        #sim.max_time_steps = max_time_steps
        #self.sim = sim
        #self.state, self.reward = get_state_reward(self.sim, self.latt, self.target_roughness)
    
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
        # if not end_flag: #check the end flag if it is true/false or 1/0.
        #     reward = -1
        # else:
        if self.reward_type=='box':
            rms_val = calc_roughness(state)
            value=rms_val - self.target_roughness
            reward=self.reward_multiplier
            reward[value>0.05*self.reward_tolerance]=-1
            reward[value<-0.05*self.reward_tolerance]=-1
        if self.reward_type=='gaussian':
            rms_val = calc_roughness(state)
            value=rms_val - self.target_roughness
            reward = gaussian(value,sig=0.04*self.reward_tolerance,mu=0)*self.reward_multiplier-1  
        if self.reward_type=='ridge':
            rms_val = calc_roughness(state)
            value=rms_val - self.target_roughness
            reward=1-np.abs(value*10/self.reward_tolerance)
            reward=reward*self.reward_multiplier
            reward[reward<-1]=-1
            #add the stuff from the notebook
        
        return state, reward, end_flag
    
    def reset(self,verbose=False):
        sim = RunSim()
        #print('Current directory is {}'.format(os.getcwd()))
        sim.read(os.path.join(self.wdir, 'kmc.input'))
        sim.init_sim()
        sim.update_rate(np.array([np.random.randint(low=1, high = 4)*self.rates_spread,
                           np.random.randint(low=1, high = 4)*self.rates_spread,
                           np.random.randint(low=1, high = 4)*self.rates_spread]), verbose=verbose)
        end_flag = sim.run_to_next_step(random_seed = np.random.randint(1,99))
        self.sim = sim
        self.state, self.reward = get_state_reward(self.sim, self.latt, self.target_roughness)
        state, reward = self.state, self.reward 
        self.end_flag=0
        return state,reward
    
    def render(self,mode='human', close=False):
        s = self.sim
        arr = s.kmc.get_conf()
        arr_1 = np.array(arr[0])
        latt = self.latt
        full_atom_box = np.zeros([latt['box'][1],latt['box'][2],latt['box'][3] ])
        for i,j,k in arr_1:
            full_atom_box[i,j,k]=1
        end_flag=self.end_flag
        surface_proj = make_surface_proj(full_atom_box)
        rms_val = calc_roughness(surface_proj)
        results = {'atom_box': full_atom_box, 'rms_val': rms_val,
'surface_proj': surface_proj, 'end_flag':end_flag}

        return results
