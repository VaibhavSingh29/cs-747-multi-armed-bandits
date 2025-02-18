"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
def bernoulli_kldiv(p, q):
    p = np.clip(p, 1e-8, 1 - 1e-8) # Clipping to avoid 0, 1 in math.log
    q = np.clip(q, 1e-8, 1 - 1e-8)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def binary_search(empirical_mean, num_arm_plays, num_plays, c=3): # Binary search for find the optimal q satisfying the KL-UCB conditon
    num_iters = 10 # Number of iterations to run search for.
    start = empirical_mean # q belongs to [empirical_mean, 1]
    end = 1
    for i in range(num_iters):
        q = (start + end) / 2 # Start at midpoint of interval and iteratively shrink interval until we exhaust num_iters.
        if (num_arm_plays * bernoulli_kldiv(empirical_mean, q)) <= (math.log(num_plays) + c * math.log(math.log(num_plays))):
            start = q
        else:
            end = q
    
    return (start + end) / 2 # Return midpoint estimate

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        
        self.num_arms = num_arms
        self.num_plays = 0 # Keeps track of the total number of plays.
        self.pull_init_arm = -1 # Helper to track arm-number for the round-robin phase
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.num_plays < self.num_arms:
            self.num_plays += 1
            self.pull_init_arm += 1
            return self.pull_init_arm # Initially we will pull each arm once, before starting the UCB loop (as per Auer et al. https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)
        else:
            best_arm = np.argmax(self.values + np.sqrt(2 * math.log(self.num_plays) / self.counts)) # UCB loop
            self.num_plays += 1
            return best_arm
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

        self.num_arms = num_arms
        self.num_plays = 0 # Keeps track of the total number of plays.
        self.pull_init_arm = -1 # Helper to track arm-number for the round-robin phase
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.num_plays < self.num_arms:
            self.num_plays += 1
            self.pull_init_arm += 1
            return self.pull_init_arm # Initially we will pull each arm once, before starting the KL-UCB loop (as per Garivier and Cappé, 2011, https://arxiv.org/abs/1102.2490)
        else:
            q_estimate = np.zeros(self.num_arms)
            for arm_idx in range(self.num_arms):
                q_estimate[arm_idx] = binary_search(self.values[arm_idx], self.counts[arm_idx], self.num_plays)
            best_arm = np.argmax(q_estimate)
            self.num_plays += 1
            return best_arm
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.alpha = np.ones(num_arms) # Setting a uniform prior for each arm.
        self.beta = np.ones(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        best_arm = np.argmax(np.random.beta(self.alpha, self.beta)) # Sample from beta for each arm, and choose best arm.
        return best_arm
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.alpha[arm_index] += reward # Update params based on reward
        self.beta[arm_index] += (1 - reward)
        # END EDITING HERE

