import numpy as np
import math, random

class EpsilonGreedy:
    def __init__(self, learner, eps, eps_dt):
        self.learner = learner
        self.eps = eps
        self.eps_dt = eps_dt

    def explore(self, observation):
        if random.random() < self.eps: # explore
            action = math.floor(random.random() * self.learner.actions)
        else: # exploit
            action = np.argmax(self.learner.Q(observation))
        return action

    def modify_eps(self):
        self.eps = max(0, self.eps + self.eps_dt)

class CountBasedOptimism(EpsilonGreedy):
    def __init__(self, learner, eps, hash_fn, exploration_bonus):
        super().__init__(learner, eps, -10 ** -4)
        self.hash_fn = hash_fn
        self.exploration_bonus = exploration_bonus
        self.counter = {}

    def explore(self, observation):
        hashed = self.hash_fn(observation)
        self.counter[hashed] = self.counter.get(hashed, 0) + 1
        bonus = self.exploration_bonus / (self.counter[hashed] ** 0.5)
        return (super().explore(observation), bonus)

class VDBE(EpsilonGreedy):
    def __init__(self, learner, eps, hash_fn, state_dim):
        super().__init__(self, learner, eps, eps_dt)
    
