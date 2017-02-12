import gym
import random
import math
import numpy as np

env = gym.make('CartPole-v0')

def multidim(dim, init = lambda: None):
    return np.random.rand(*dim)

def multidex(iterable, index):
    return iterable[index]

def set_multidex(iterable, index, value):
    iterable[index] = value

argmax = lambda pairs: max(pairs, key=lambda x: x[1])[0]
argmax_index = lambda values: argmax(enumerate(values))

class TabularLearner:
    BUCKETS = (50, 200, 50, 200)
    LIMITS = (4.8, 30, 0.42, 30)
    DISCOUNT = 0.95
    ALPHA_MIN = 0.05

    def __init__(self, actions, buckets = BUCKETS):
        self.Q = multidim(buckets + (actions,), lambda: random.random())
        self.eps = 0.3
        self.alpha = 0.5
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

    def learn(self, observation, reward, done):
        a_t, s_t = self.last
        prev = multidex(self.Q, self.discretized(s_t) + (a_t,))
        now = multidex(self.Q, self.discretized(observation))
        update = prev + max(TabularQLearner.ALPHA_MIN, self.alpha) * (reward + TabularQLearner.DISCOUNT * max(now) - prev)
        if done: update = reward
        set_multidex(self.Q, self.discretized(s_t) + (a_t,), update)

    def discretized(self, observation):
        bounded = [min(self.LIMITS[i],max(-self.LIMITS[i], observation[i])) for i in range(len(observation))]
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


learner = TabularSARSALearner(env.action_space.n)


def main():
    i_episode = 0
    while True:
        i_episode += 1
        observation = env.reset()
        for t in range(1000):
            if i_episode % 10000 == 0 or i_episode > 100000: env.render()
            action = learner.act(observation)
            observation, reward, done, info = env.step(action)
            learner.learn(observation, reward, done)
            if done:
                print("Episode {0:8d}: {1:4d} timesteps".format(i_episode, t+1))
                break
            
main()

