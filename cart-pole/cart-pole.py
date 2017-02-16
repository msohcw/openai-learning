import gym
from gym import wrappers
import random
import math
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Dense

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, "/tmp/cartpole0", force = True)

"""

def multidim(dim, init = lambda: None):
    return np.random.rand(*dim)
"""

def multidex(iterable, index):
    return iterable[index]

def set_multidex(iterable, index, value):
    iterable[index] = value

argmax = lambda pairs: max(pairs, key=lambda x: x[1])[0]
argmax_index = lambda values: argmax(enumerate(values))

class TabularLearner:
    BUCKETS = (15, 15, 10, 10)
    LIMITS = (4.8, 10, 0.42, 5)
    DISCOUNT = 0.95
    ALPHA_MIN = 0.05
    TIMESTEP_MAX = 200

    def __init__(self, actions, buckets = BUCKETS):
        self.Q = np.random.rand(*buckets, actions)
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
    
    def learn(*args):
        pass

    def discretized(self, observation):
        eps = 0.000001
        bounded = [min(self.LIMITS[i] - eps,max(-self.LIMITS[i] + eps, observation[i])) for i in range(len(observation))]
        return tuple(math.floor((bounded[i] + self.LIMITS[i]) / 2 / self.LIMITS[i] * self.BUCKETS[i]) for i in range(len(bounded)))
        
class TabularQLearner(TabularLearner):
    def learn(self, s0, observation, reward, action, done, t):
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

class TabularSARSALearner(TabularLearner):
    def learn(self, observation, reward, done):
        a_t, s_t = self.last
        prev = multidex(self.Q, self.discretized(s_t) + (a_t,))
        now = multidex(self.Q, self.discretized(observation))
        expected = max(now) * (1 - self.eps) + self.eps * sum(now)/len(now)
        update = prev + max(TabularQLearner.ALPHA_MIN, self.alpha) * (reward + TabularQLearner.DISCOUNT * expected - prev)
        if done: update = reward
        set_multidex(self.Q, self.discretized(s_t) + (a_t,), update)

class DeepQLearner:
    MINIMUM_EXPERIENCE = 200

    def __init__(self, input_dim, layers, batch_size, total_memory):
        self.total_memory = total_memory
        self.batch_size = batch_size
        # experience has s0, action, reward, s1, and done? and t
        self.experience = np.zeros((input_dim * 2 + 4, total_memory))
        self.exp_ct = 0
        self.exp_index = 0
        self.model = Sequential()
        self.input_dim = input_dim

        first, layers, end = layers[0], layers[1:-1], layers[-1]
        self.model.add(Dense(first, 
            batch_input_shape=(None, input_dim), 
            init = 'uniform', 
            activation = 'relu'))
        for layer in layers:
            self.model.add(Dense(layer, activation = 'relu', init = 'uniform'))
        # end with linear to sum to Q-value
        self.model.add(Dense(end, activation = 'linear', init = 'uniform'))
        rms = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer = rms, loss = 'mse')

    def learn(self, s0, s1, reward, action, done, t):
        memory = np.vstack((
                np.array(s0).reshape(len(s0), 1),
                np.array(s1).reshape(len(s1), 1),
                reward,
                action,
                done,
                t
                ))
        self.experience[:, self.exp_index] = memory.flatten()
        if self.exp_ct < self.total_memory: self.exp_ct += 1
        self.exp_index += 1
        self.exp_index %= self.total_memory

        if self.exp_ct > DeepQLearner.MINIMUM_EXPERIENCE: 
            self.experience_replay()

    def experience_replay(self):
        subset_index = np.random.choice(self.exp_ct, self.batch_size, replace = False)


        subset = self.experience[:, subset_index]
        s0 = subset[:self.input_dim, :]
        s1 = subset[self.input_dim:self.input_dim*2, :]
        r = subset[self.input_dim*2, :]
        action = subset[self.input_dim*2+1, :]
        done = subset[self.input_dim*2+2, :]
        t = subset[self.input_dim*2+3, :]

        s0_q_values = self.model.predict(s0.T) 
        s1_q_values = np.amax(self.model.predict(s1.T), axis = 1) # take maximum future

        #print(s0_q_values)

        # update Q values
        for k, q in enumerate(s0_q_values):
            a = int(action[k])
            if bool(done[k]):
                if t == 200:
                    q[a] = 10
                else:
                    q[a] = -10
            else:
                q[a] = r[k] + 0.95 * s1_q_values[k]
        
        loss = self.model.train_on_batch(s0.T, s0_q_values)

    def act(self, observation):
        q = self.model.predict(observation.reshape(1,4))
        return np.argmax(q)

# there's a bug with the unmonitored envs not checking max_steps
max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

def main():
    total = 0
    #learner = DeepQLearner(len(env.observation_space.high), (8, 16, 32, env.action_space.n), 128, 10000)
    learner = TabularQLearner(env.action_space.n)
    for i_episode in range(1000):
        observation = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            action = learner.act(observation)
            s0 = observation
            observation, reward, done, info = env.step(action)
            learner.learn(s0, observation, reward, action, done, t+1)
            ep_reward += reward
            if done: break
        total += ep_reward
        print("Episode {0:8d}: {1:4d} timesteps, {2:4f} average".format(i_episode, t+1, total/(i_episode+1)))
    env.close()
    gym.upload('/tmp/cartpole0', api_key='sk_Q5yxeYioS96EjTbGnXtWxA')
main()


