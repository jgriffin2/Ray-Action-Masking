import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from typing import Optional

""" 
    Simple environment for testing masking.
    This environment simply increments a step number, then
    masks all actions except for the step number.
    
    As a result, only one action should be specified by the action-masking
    RL module.  
    
    Print a note about entries into each method for monitoring.
    
    Success is denoted by following the print statements in the step method.
    It shows that only the unmasked action is returned from the RLModule
"""    

class TestActionMask(gym.Env):

    def __init__(self, config):

        print ('In environment __init__()')
        self.config = config
        num_param =  10
               
        
        low = np.zeros((num_param,))
        high = np.ones((num_param,)) * 9.
                 
        self.observations = Box(low, high)                           

        self.action_space = Discrete(10)
                
        self.observation_space = Dict(
            {
                "action_mask":  Box(0.0, 1.0, shape=(self.action_space.n,)),
                "observations":    self.observations
            }
        )
        self.value = np.arange(0, 10, dtype=np.float32)
        self.action_mask = np.zeros((10,))
    
        self.stepno = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        print ('In environment reset()')
        self.stepno = 0
    
        self.action_mask = np.zeros((10,))
        self.action_mask[self.stepno] = 1
        obs = {'action_mask': self.action_mask,
               'observations': self.value}
        info = {'step':self.stepno}
        return obs, info
    
    def step(self, action):
        print (f'In environment step, action={action}')
        reward = self.value[action]
        
        self.stepno += 1
            
        self.action_mask = np.zeros((10,))
        self.action_mask[self.stepno] = 1
        obs = {'action_mask': self.action_mask,
               'observations': self.value}
        info = {'step': self.stepno}
    
        if self.stepno == 9:
            terminated = True
            truncated = True
        else:
            terminated = False
            truncated = False
                    
        return obs, reward, terminated, truncated, info
