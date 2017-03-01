import random
import math
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

EPSILON = 1 * 10 ** -6

class TabularLearner:
    def __init__(self, actions, buckets, limits, terminal_fn, 
                        discount = 0.95, eps = 1, alpha = 0.5, alpha_min = 0.1, eps_dt = -10 ** -5, alpha_dt = -10 ** -4):
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
        self.eps = max(0, self.eps + self.eps_dt)
        return action
    
    def learn(self, s0, s1, reward, action, done, t):
        prev = self.Q[self.discretized(s0) + (action,)]
        now = self.Q[self.discretized(s1)]
        if done: # update should be terminal state update
            update = self.terminal_fn(reward, t)
        else: # update as per normal
            update = self.update_fn(prev, now, reward)
        self.Q[self.discretized(s0) + (action,)] = update
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
                    eps = 0.5, eps_dt = -10 * 10 ** -5, discount = 0.9, target_freeze_duration = 2500, rgb_array = False):
        self.total_memory = total_memory
        self.batch_size = batch_size
        self.terminal_fn = terminal_fn

        if not rgb_array:
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

        self.build_model(input_dim, layers, rgb_array)

        rms = keras.optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer = rms, loss = 'mse')
        self.target_model.compile(optimizer = rms, loss = 'mse')

    def build_model(self, input_dim, layers, rgb_array):
        first, layers, end = layers[0], layers[1:-1], layers[-1]
        
        if not rgb_array:
            first_layer = Dense(first,
                                batch_input_shape = (None, input_dim),
                                init = 'uniform',
                                activation = 'relu')
            self.model.add(first_layer)
            self.target_model.add(first_layer)
        else:
            first_conv = Convolution2D(128, 3, 3, border_mode = 'same', input_shape = input_dim, dim_ordering='th')
            first_pool = MaxPooling2D(pool_size=(2, 2)) 
            second_conv = Convolution2D(64, 3, 3, border_mode = 'same')
            second_pool = MaxPooling2D(pool_size=(2, 2)) 
            flatten = Flatten()
            conv_layers = [first_conv, first_pool, second_conv, second_pool, flatten]
            for layer in conv_layers:
                self.model.add(layer)
                self.target_model.add(layer)
        
        for layer in layers:
            l = Dense(layer, activation = 'relu', init = 'uniform')
            self.model.add(l)
            self.target_model.add(l)
            
        # end with linear to sum to Q-value
        end_layer = Dense(end, activation = 'linear', init = 'uniform')
        self.model.add(end_layer)
        self.target_model.add(end_layer)

    def learn(self, s0, s1, reward, action, done, t):
        self.store_experience(s0, s1, reward, action, done, t)

        if self.exp_ct < self.total_memory: self.exp_ct += 1
        self.exp_index += 1
        self.exp_index %= self.total_memory

        if self.exp_index % self.target_freeze_duration == 0:
            print("Copied to target network")
            self.target_model.set_weights(self.model.get_weights())

        if self.exp_ct > DeepQLearner.MINIMUM_EXPERIENCE: 
            self.experience_replay()

    def store_experience(self, s0, s1, reward, action, done, t):
        memory = np.vstack((
                np.array(s0).reshape(len(s0), 1),
                np.array(s1).reshape(len(s1), 1),
                reward,
                action,
                done,
                t
                ))
        self.experience[:, self.exp_index] = memory.flatten()

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
                q[a] = self.terminal_fn(r[k], t[k])
            else:
                q[a] = r[k] + self.discount * s1_q_values[k]
        
        loss = self.model.train_on_batch(s0.T, s0_q_values)

    def future_fn(self, s1):
        return np.amax(self.target_model.predict(s1.T), axis = 1)

    def act(self, observation):
        observation = observation.reshape(1, *observation.shape)

        q = self.model.predict(observation)
        if random.random() < self.eps:
            action = math.floor(random.random() * self.actions)
        else:
            action = np.argmax(q)
        self.eps = max(0, self.eps + self.eps_dt)
        return action

class DoubleDeepQLearner(DeepQLearner):
    def future_fn(self, s1):
        # use model to pick actions, target to evaluate Q(s, a)
        actions = np.argmax(self.model.predict(s1), axis = 1)
        q_values = self.target_model.predict(s1)
        # I couldn't find a better numpy way to do this,
        # it takes the element of each row according to actions
        return np.diagonal(np.take(q_values, actions, axis = 1))

class PixelDataDoubleDeepQLearner(DoubleDeepQLearner):
    def __init__(self, input_dim, *args, **kwargs):
        super(PixelDataDoubleDeepQLearner, self).__init__(input_dim, *args, **kwargs)
        doubled = (input_dim[0] * 2,) + input_dim[1:]
        # change experience memory to s0, s1 pixel array
        self.experience = np.zeros((self.total_memory, *doubled))
        self.transition_data = np.zeros((self.total_memory, 4)) # reward, action, done, t
    
    def future_fn(self, s1):
        # use model to pick actions, target to evaluate Q(s, a)
        actions = np.argmax(self.model.predict(s1), axis = 1)
        q_values = self.target_model.predict(s1)
        # I couldn't find a better numpy way to do this,
        # it takes the element of each row according to actions
        return np.diagonal(np.take(q_values, actions, axis = 1))
    
    def store_experience(self, s0, s1, reward, action, done, t):

        self.experience[self.exp_index, ...] = np.vstack((s0, s1))
        self.transition_data[self.exp_index, :] = np.array((reward, action, done, t)).flatten()

    def experience_replay(self):
        #print("Replaying experiences...", end = " ")
        subset_index = np.random.choice(self.exp_ct, self.batch_size, replace = False)
        subset = self.experience[subset_index, ...] # Ellipsis!
        subset_transitions = self.transition_data[subset_index, :]

        s0 = subset[:, :self.input_dim[0], ...]
        s1 = subset[:, self.input_dim[0]:, ...]

        r, action, done, t = (subset_transitions[:, i] for i in range(4))

        s0_q_values = self.model.predict(s0) 
        s1_q_values = self.future_fn(s1)

        # update Q values
        for k, q in enumerate(s0_q_values):
            a = int(action[k])
            if bool(done[k]):
                q[a] = self.terminal_fn(r[k], t[k])
            else:
                q[a] = r[k] + self.discount * s1_q_values[k]
        
        loss = self.model.train_on_batch(s0, s0_q_values)
        #print("completed.")
