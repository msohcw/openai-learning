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

class VDBE(EpsilonGreedy):
    pass
