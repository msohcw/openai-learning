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
EPSILON = 1 * 10 ** -6

class TabularLearner:
    def __init__(self, actions, buckets, limits, terminal_fn, 
                        discount = 0.95, eps = 0.3, alpha = 0.5, alpha_min = 0.1, eps_dt = -10 ** -5, alpha_dt = -10 ** -4):
        self.Q = np.random.rand(*buckets, actions)
        self.actions = actions
        self.buckets = buckets
        self.limits = limits
        self.terminal_fn = terminal_fn

        self.discount = discount
        self.eps = eps
        self.eps_dt = eps_dt
        self.alpha = alpha
        self.alpha_dt = alpha_dt
        self.alpha_min = alpha_min

    def act(self, observation):
        if random.random() < self.eps: # explore
            action = math.floor(random.random() * self.actions)
        else: # exploit
            action = np.argmax(self.Q[self.discretized(observation)])
        self.last = (action, observation)
        self.eps = max(0, self.eps + self.eps_dt)
        return action
    
    def learn(self, s0, s1, reward, action, done, t):
        a_t, s_t = self.last
        prev = self.Q[self.discretized(s_t) + (a_t,)]
        now = self.Q[self.discretized(s1)]
        if done: # update should be terminal state update
            update = self.terminal_fn(reward, t)
        else: # update as per normal
            update = self.update_fn(prev, now, reward)
        self.Q[self.discretized(s_t) + (a_t,)] = update
        self.alpha = max(self.alpha_min, self.alpha + self.alpha_dt)

    def update_fn(self, q0_a, q1, r):
        return max(q1) # default, bad update

    def discretized(self, observation):
        b, l = self.buckets, self.limits # shorthand
        # EPSILON used to keep within bucket bounds
        bounded = [min(l[i] - EPSILON,max(-l[i] + EPSILON, observation[i])) for i in range(len(observation))]
        return tuple(math.floor((bounded[i] + l[i]) / 2 / l[i] * b[i]) for i in range(len(bounded)))
        
class TabularQLearner(TabularLearner):
    def update_fn(self, q0_a, q1, r):
        return q0_a + max(self.alpha_min, self.alpha) * (r + self.discount * max(q1) - q0_a)

class TabularSARSALearner(TabularLearner):
    def update_fn(self, q0_a, q1, r):
        expected = max(q1) * (1 - self.eps) + self.eps * sum(q1) / len (q1)
        return q0_a + max(self.alpha_min, self.alpha) * (r + self.discount * expected - q0_a)

class DeepQLearner:
    MINIMUM_EXPERIENCE = 1000

    def __init__(self, input_dim, layers, batch_size, total_memory, terminal_fn,
                    eps = 0.1, eps_dt = -10 * 10 ** -5, discount = 0.95, target_freeze_duration = 2500):
        self.total_memory = total_memory
        self.batch_size = batch_size
        self.terminal_fn = terminal_fn

        # experience has s0, s1, action, reward, done? and t
        self.experience = np.zeros((input_dim * 2 + 4, total_memory))
        self.exp_ct = 0
        self.exp_index = 0
        self.model = Sequential()
        self.target_model = Sequential()
        self.input_dim = input_dim
        self.actions = layers[-1]

        self.eps = eps
        self.eps_dt = eps_dt
        self.discount = discount
        self.target_freeze_duration = target_freeze_duration

        first, layers, end = layers[0], layers[1:-1], layers[-1]
        
        first_layer = Dense(first,
                            batch_input_shape = (None, input_dim),
                            init = 'uniform',
                            activation = 'relu')

        self.model.add(first_layer)
        self.target_model.add(first_layer)
        
        for layer in layers:
            l = Dense(layer, activation = 'relu', init = 'uniform')
            self.model.add(l)
            self.target_model.add(l)
            
        # end with linear to sum to Q-value
        end_layer = Dense(end, activation = 'linear', init = 'uniform')
        self.model.add(end_layer)
        self.target_model.add(end_layer)

        rms = keras.optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer = rms, loss = 'mse')
        self.target_model.compile(optimizer = rms, loss = 'mse')

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

        if self.exp_index % self.target_freeze_duration == 0:
            print("Copied to target network")
            self.target_model.set_weights(self.model.get_weights())

        if self.exp_ct > DeepQLearner.MINIMUM_EXPERIENCE: 
            self.experience_replay()

    def experience_replay(self):
        subset_index = np.random.choice(self.exp_ct, self.batch_size, replace = False)

        # this hideous mess encodes an experience
        subset = self.experience[:, subset_index]
        s0 = subset[:self.input_dim, :]
        s1 = subset[self.input_dim:self.input_dim*2, :]
        r = subset[self.input_dim*2, :]
        action = subset[self.input_dim*2+1, :]
        done = subset[self.input_dim*2+2, :]
        t = subset[self.input_dim*2+3, :]

        s0_q_values = self.model.predict(s0.T) 
        s1_q_values = self.future_fn(s1)

        # update Q values
        for k, q in enumerate(s0_q_values):
            a = int(action[k])
            if bool(done[k]):
                q[a] = self.terminal_fn(r, t[k])
            else:
                q[a] = r[k] + self.discount * s1_q_values[k]
        
        loss = self.model.train_on_batch(s0.T, s0_q_values)

    def future_fn(self, s1):
        return np.amax(self.target_model.predict(s1.T), axis = 1)

    def act(self, observation):
        q = self.model.predict(observation.reshape(1,4))
        if random.random() < self.eps:
            action = math.floor(random.random() * self.actions)
        else:
            action = np.argmax(q)
        self.eps = max(0, self.eps + self.eps_dt)
        return action

class DoubleDeepQLearner(DeepQLearner):
    def future_fn(self, s1):
        # use model to pick actions, target to evaluate Q(s, a)
        actions = np.argmax(self.model.predict(s1.T), axis = 1)
        q_values = self.target_model.predict(s1.T)
        # I couldn't find a better numpy way to do this,
        # it takes the element of each row according to actions
        return np.diagonal(np.take(q_values, actions, axis = 1))

# there's a bug with the unmonitored envs not checking max_steps
max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

learner = None

def main():
    global learner
    BUCKETS = (15, 15, 10, 10) # buckets to discretize
    LIMITS = (4.8, 10, 0.42, 5) # limits to discretize
    TIMESTEP_MAX = max_steps

    # no_drop is a reward function that penalises falling before 200. accelerates learning massively.
    no_drop = lambda fail, success: lambda r, t: fail if t != TIMESTEP_MAX else success

    learner = DoubleDeepQLearner(len(env.observation_space.high), (8, 16, 32, env.action_space.n), 256, 100000, no_drop(-10, 10))
    #learner = DeepQLearner(len(env.observation_space.high), (8, 16, 32, env.action_space.n), 256, 100000, no_drop(-10, 10))
    #learner = TabularQLearner(env.action_space.n, BUCKETS, LIMITS, no_drop(-200, 10))
    #learner = TabularSARSALearner(env.action_space.n, BUCKETS, LIMITS, no_drop(-200,10))

    total = 0
    for i_episode in range(1000):
        s1 = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            action = learner.act(s1)
            s0 = s1
            s1, reward, done, info = env.step(action)
            learner.learn(s0, s1, reward, action, done, t+1)
            ep_reward += reward
            if done: break
        total += ep_reward
        print("Episode {0:8d}: {1:4d} timesteps, {2:4f} average".format(i_episode, t+1, total/(i_episode+1)))
    env.close()
    # uncomment this line with your api key if you want to upload to openai
    #gym.upload('/tmp/cartpole0', api_key='')
main()


