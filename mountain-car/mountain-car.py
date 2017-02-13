import gym
from gym import wrappers
import random
import math
import numpy as np
import os

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, "/tmp/cartpole0", force = True)

def multidim(dim, init = lambda: None):
    return np.random.rand(*dim)

def multidex(iterable, index):
    return iterable[index]

def set_multidex(iterable, index, value):
    iterable[index] = value

argmax = lambda pairs: max(pairs, key=lambda x: x[1])[0]
argmax_index = lambda values: argmax(enumerate(values))

class TabularLearner:
    BUCKETS = (15, 15, 10, 10)
    LIMITS = (4.8, 10, 0.42, 5)
    #BUCKETS = (10, 10)
    #LIMITS = (1.2, 0.07)
    DISCOUNT = 0.95
    ALPHA_MIN = 0.05
    TIMESTEP_MAX = 200

    def __init__(self, actions, buckets = BUCKETS):
        self.Q = multidim(buckets + (actions,), lambda: random.random())
        self.eps = 0.3
        self.alpha = 1
        self.actions = actions

    def act(self, observation):
        if random.random() < self.eps:
            action = math.floor(random.random() * self.actions)
        else:
            action = argmax_index(multidex(self.Q, self.discretized(observation)))
        self.last = (action, observation)
        self.eps -= 0.00001
        self.eps = max(0, self.eps)
        self.alpha -= 0.0001
        return action

    def learn(self, observation, reward, done, t):
        a_t, s_t = self.last
        prev = multidex(self.Q, self.discretized(s_t) + (a_t,))
        now = multidex(self.Q, self.discretized(observation))
        update = prev + max(TabularQLearner.ALPHA_MIN, self.alpha) * (reward + TabularQLearner.DISCOUNT * max(now) - prev)
        if done:
            if t != TabularLearner.TIMESTEP_MAX: 
                update = -TabularLearner.TIMESTEP_MAX
            else:
                update = 100
        set_multidex(self.Q, self.discretized(s_t) + (a_t,), update)

    def discretized(self, observation):
        eps = 0.000001
        bounded = [min(self.LIMITS[i] - eps,max(-self.LIMITS[i] + eps, observation[i])) for i in range(len(observation))]
        return tuple(math.floor((bounded[i] + self.LIMITS[i]) / 2 / self.LIMITS[i] * self.BUCKETS[i]) for i in range(len(bounded)))
        
class TabularQLearner(TabularLearner):
    pass

class TabularSARSALearner(TabularLearner):
    def learn(self, observation, reward, done):
        a_t, s_t = self.last
        prev = multidex(self.Q, self.discretized(s_t) + (a_t,))
        now = multidex(self.Q, self.discretized(observation))
        expected = max(now) * (1 - self.eps) + self.eps * sum(now)/len(now)
        update = prev + max(TabularQLearner.ALPHA_MIN, self.alpha) * (reward + TabularQLearner.DISCOUNT * expected - prev)
        if done: update = reward
        set_multidex(self.Q, self.discretized(s_t) + (a_t,), update)

max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
def main():
    total = 0
    learner = TabularQLearner(env.action_space.n)
    for i_episode in range(1, 3001):
        observation = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            action = learner.act(observation)
            observation, reward, done, info = env.step(action)
            learner.learn(observation, reward, done, t+1)
            ep_reward += reward
            if done: break
        total += ep_reward
        print("Episode {0:8d}: {1:4d} timesteps, {2:4f} average".format(i_episode, t+1, total/i_episode))
    env.close()
    gym.upload('/tmp/cartpole0', api_key='sk_Q5yxeYioS96EjTbGnXtWxA')
main()

