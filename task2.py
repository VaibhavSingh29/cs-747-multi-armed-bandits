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
        # END EDITING HERE
    
    def give_query_set(self):
        # START EDITING HERE
        if self.num_plays < self.num_arms:
            self.num_plays += 1
            self.pull_init_arm += 1
            return [self.pull_init_arm]
        else:
            query_set = []
            best_arm = np.argmax(self.values + np.sqrt(2 * math.log(self.num_plays) / self.counts))
            query_set.append(best_arm)
            arm_set = [i for i in range(self.num_arms) if i not in query_set]
            for i in range(self.k):
                query_set.append(np.random.choice(arm_set, replace=False))
            self.num_plays += 1
            return query_set
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        #END EDITING HERE

