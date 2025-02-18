"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the CostlySetBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_query_set(self): This method is called when the algorithm needs to
        provide a query set to the oracle. The method should return an array of 
        arm indices that specifies the query set.
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_query_set method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
"""

import numpy as np
from task1 import Algorithm
# START EDITING HERE
# You can use this space to define any helper functions that you need
import math
# END EDITING HERE

class CostlySetBanditsAlgo(Algorithm):
    def __init__(self, num_arms, horizon):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        # START EDITING HERE
        self.counts = np.zeros(self.num_arms)
        self.values = np.zeros(self.num_arms)
        self.alpha = np.ones(num_arms) # Setting a uniform prior for each arm.
        self.beta = np.ones(num_arms)
        self.k = 0.5 # Thresholding hparam
        self.query_set = None
        # END EDITING HERE
    
    def give_query_set(self):
        # START EDITING HERE
        samples = np.random.beta(self.alpha, self.beta)
        var = (self.alpha * self.beta) / ((self.alpha + self.beta + 1) * (self.alpha + self.beta) ** 2)
        mean = (self.alpha) / (self.alpha + self.beta)
        self.query_set = [i for i, s in enumerate(samples) if s >= self.k] # Select arms where sample from beta is greater or equal to 0.5 for exploitation
        self.query_set.extend([i for i, s in enumerate(samples) if var[i] > 1e-2]) # We also select arms with high variance for exploration
        self.query_set = list(set(self.query_set))
        if len(self.query_set) == 0:
            self.query_set = np.argsort(-samples)[:self.num_arms // 2].tolist() # In case, query set is empty, we take top-k arms where k = num_arms / 2
        return self.query_set
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.alpha[arm_index] += reward # Update params based on reward
        self.beta[arm_index] += (1 - reward)
        #END EDITING HERE

