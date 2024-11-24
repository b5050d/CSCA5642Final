"""
First attempt at carpole

Going to try to make this into my final

# TODO - before running, pre-emptively map out the Decay of the epsilon over
epis to make sure it makes sense

# TODO - add better decay
# TODO - add memory limit before training
# TODO - add batching of training
# TODO - add different models - starting with a big boy model

# TODO - if the model is the best so far, save it
    # The reason for this is because we jsut want to see what its doing

"""

import os
import gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
from matplotlib import pyplot as plt

class CartPoleModel():
    """
    
    """

    def __init__(self):
        """
        Set up the class
        """
        self.env = gym.make("CartPole-v1", render_mode = 'rgb_array')
        self._define_constants()

        self.build_model_0()

    def _define_constants(self):
        """
        Define Constants, variables
        """
        self.RANDOM_SEED = 23
        np.random.seed(self.RANDOM_SEED)
        random.seed(self.RANDOM_SEED)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.N_EPISODES = 1000
        self.MIN_EPSILON = .001
        self.EPSILON_DECAY_RATE = .999
        self.epsilon = 1

        self.BATCH_SIZE = 128
        self.LEARNING_RATE = .0001
        self.GAMMA = .99

        # Memory
        self.memory = []
        self.MAX_MEMORY_SIZE = 2000
        self.MIN_MEMORY_SIZE = 1000

        self.EVALUATION_BATCH = 20

        # Debugging stuff
        currdir = os.path.dirname(__file__)
        self.results_table = []
        self.csv_dump = currdir + "\\test.csv"

    def preliminary_epsilon_plot(self):
        xs = [i for i in range(self.N_EPISODES*100)]
        ys = []
        ep = self.epsilon
        for x in xs:
            ep = ep * self.EPSILON_DECAY_RATE
            ys.append(ep)
        plt.plot(xs, ys)
        plt.xlabel("# of Traning Batches")
        plt.ylabel("Epsilon (Explore Rate)")
        plt.title("Epsilon Decay over planned episodes")
        plt.grid()
        plt.pause(1)
        plt.cla()

    def append_to_results_table(self, episode, reward, epsilon):
        d = {}
        d["episode"] = episode 
        d["reward"] = int(reward)
        d["epsilon"] = epsilon
        self.results_table.append(d)

        df = pd.DataFrame(self.results_table)
        df.to_csv(self.csv_dump)

    def build_model_0(self):
        """
        """
        self.model = Sequential()
        self.model.add(Dense(512, input_dim = self.n_states, activation = 'relu'))
        self.model.add(Dense(256, activation = 'relu'))
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dense(self.n_actions, activation = 'linear'))
        opt = Adam(learning_rate = self.LEARNING_RATE) 
        self.model.compile(
            loss='mse',
            optimizer = opt)
        
    def perform_action(self, state):
        """
        go ahead and take action, this requires a model first to use
        """
        if np.random.rand() <= self.epsilon:
            # Exploring
            return self.env.action_space.sample()
        else:
            # Going with Model's best guess so far
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values[0])

    def decay_epsilon(self):
        """
        Decay the Epsilon Value
        """
        self.epsilon = round(max(self.epsilon * self.EPSILON_DECAY_RATE, self.MIN_EPSILON),4)
    
    def store_to_memory(self, state, action, reward, next_state, stop_condition):
        """
        Store experiences into memory for recall and use in batch training later
        """
        
        self.memory.append((
            state,
            action,
            reward,
            next_state,
            stop_condition,
        ))

        # Keep the memory under the maximum size
        if len(self.memory) > self.MAX_MEMORY_SIZE:
            self.memory.pop(0)
        
    def learn_by_batch(self):
        """
        Pull experiences from the memory and apply learning
        """
        if len(self.memory) < self.BATCH_SIZE:
            # There are not enough datas to learn from yet
            print("There are not yet enough samples to learn from")
            return
        
        # print("Initiating Batch Learning")
        batch = random.sample(self.memory, self.BATCH_SIZE)

        states = np.zeros((self.BATCH_SIZE, self.n_states))
        next_states = np.zeros((self.BATCH_SIZE, self.n_states))
        
        actions = []
        rewards = []
        stop_conditions = []

        for iter, memory_slice in enumerate(batch):
            states[iter] = memory_slice[0][0]
            actions.append(memory_slice[1])
            rewards.append(memory_slice[2])
            next_states[iter] = memory_slice[3][0]
            stop_conditions.append(memory_slice[4])

        q_values = self.model.predict(states, verbose=0)
        q_values_next = self.model.predict(next_states, verbose = 0)
    
        for i in range(len(states)):
            if stop_conditions[i] == True:
                q_values[i][actions[i]] = rewards[i]

            else:
                # Bellman equation
                q_values[i][actions[i]] = rewards[i] + self.GAMMA * (np.amax(q_values_next[i]))

        # Fit the model given the new desired outcome
        self.model.fit(states, q_values, verbose = 0)

    def test_loop(self):
        print("starting test loop")

    def perform_step(self, action):
        """
        Placing this method here to maximize code re-use
        Just parses the step response
        """
        ans = self.env.step(action)
        next_state = np.reshape(ans[0], [1, self.n_states])
        reward = ans[1]
        stop_condition = ans[2]
        return next_state, reward, stop_condition

    def main_loop(self):
        rewards = []
        episodes = []

        print("========================================")
        print("Performing Initial Memory Population")
        while len(self.memory) < self.MIN_MEMORY_SIZE:
            state = self.env.reset()
            state = np.reshape(state[0], [1,self.n_states])
            stop_condition = False

            while stop_condition is not True:
                action = self.env.action_space.sample()
                next_state, reward, stop_condition = self.perform_step(action)
                self.store_to_memory(state, action, reward, next_state, stop_condition)
        print("Finished Initial Memory Population")
        print("========================================")
        print("")

        print("========================================")
        print("Beginning Training Loop")
        best_reward = 0
        for episode in range(self.N_EPISODES):
            state = self.env.reset()
            state = np.reshape(state[0], [1,self.n_states])

            total_reward = 0

            stop_condition = False
            while stop_condition is not True:
                action = self.perform_action(state)
                next_state, reward, stop_condition = self.perform_step(action)
                self.store_to_memory(state, action, reward, next_state, stop_condition)

                state = next_state
                total_reward += reward

                self.learn_by_batch()
                self.decay_epsilon()

            self.append_to_results_table(episode, total_reward, self.epsilon)

            print("Episode {}, Reward: {}, Epsilon: {}".format(episode, int(total_reward), self.epsilon))

        print("========================================")
        print("")

    def test_loop(self):
        state = self.env.reset()
        state = np.reshape(state[0], [1,self.n_states])
        print(state)
        print(state.shape)
        
if __name__=="__main__":
    a = CartPoleModel()
    a.test_loop()
    # a.main_loop()