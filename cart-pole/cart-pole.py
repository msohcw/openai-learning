import gym
import random
import math

env = gym.make('CartPole-v0')

def multidim(dim, init = lambda: None):
    if dim == (): return init()
    return [multidim(dim[1:], init) for _ in range(dim[0])]

def multidex(iterable, index):
    if len(index) == 1:
        return iterable[index[0]]
    return multidex(iterable[index[0]], index[1:])

def set_multidex(iterable, index, value):
    if len(index) == 1:
        iterable[index[0]] = value;
    else:
        set_multidex(iterable[index[0]], index[1:], value)


argmax = lambda pairs: max(pairs, key=lambda x: x[1])[0]
argmax_index = lambda values: argmax(enumerate(values))

class TabularQLearner:
    BUCKETS = (50, 100, 50, 100)
    LIMITS = (4.8, 20, 0.42, 20)
    ALPHA = 0.15
    DISCOUNT = 0.95

    def __init__(self, actions, buckets = BUCKETS):
        self.Q = multidim(buckets + (actions,), lambda: random.random())
        self.eps = 0.3
        self.actions = actions

    def act(self, observation):
        if random.random() < self.eps:
            action = math.floor(random.random() * self.actions)
        else:
            action = argmax_index(multidex(self.Q, self.discretized(observation)))
        self.last = (action, observation)
        self.eps -= 0.00001
        return action

    def learn(self, observation, reward, done):
        a_t, s_t = self.last
        prev = multidex(self.Q, self.discretized(s_t) + (a_t,))
        now = multidex(self.Q, self.discretized(observation))
        update = prev + TabularQLearner.ALPHA * (reward + TabularQLearner.DISCOUNT * max(now) - prev)
        if done: update = reward
        set_multidex(self.Q, self.discretized(s_t) + (a_t,), update)

    def discretized(self, observation):
        bounded = [min(self.LIMITS[i],max(-self.LIMITS[i], observation[i])) for i in range(len(observation))]
        return tuple(math.floor((bounded[i] + self.LIMITS[i]) / 2 / self.LIMITS[i] * self.BUCKETS[i]) for i in range(len(bounded)))
        
learner = TabularQLearner(env.action_space.n)

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

