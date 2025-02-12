from colorama import init
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random
import os
from sphere import Sphere
import copy

class RubiksCubeSolver:
    def __init__(self):
        self.model = self._build_model()
        self.sphere = Sphere()
        self.epsilon = 0.5 # Exploration rate
        self.memory_queue = [] # Queue of states
        self.history = [] #array of states and rewards
        self.history_folder = 'history'
        self.model_save_path = 'models/simple_dense_1.keras'
        self.action_range = 12
        self.model_sequence_length = 3
        self.epochs = 100
        self.color_to_int={
            'green':0,
            'purple':1,
            'red':2,
            'blue':3,
            'yellow':4,
            'orange':5,
        }

    def save_history(self):
        filepath = f'{self.history_path}/history_{len(os.listdir(self.history_path))}.npy'
        np.save(filepath, self.history)
    def load_history(self):
        for file in os.listdir(self.history_folder):
            self.history.append(np.load(file))
    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(6, 9)))#keep faces together
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(12))  # Output layer with 12 logits
        model.compile(optimizer='adam', loss='mse')
        return model

    def infer(self, state):
        if random.random() < self.epsilon:
            return [random.randint(0, self.action_range)]
        # state = np.array([[self.color_to_int[row] for row in s] for s in state])
        state = self._reformat_state(state=state)
        state = np.array(state).reshape((1, 6, 9))
        logits = self.model.predict(state)

        #add to memory queue
        while len(self.memory_queue) > 5:
            self.memory_queue.pop(0)
        rewards=[]
        for action in range(self.action_range):
            # sphere_copy = copy.deepcopy(self.sphere)
            sphere_copy = Sphere()
            sphere_copy.points = copy.deepcopy(self.sphere.points)
            sphere_copy.move(action)
            rewards.append(self.get_reward_from_state(sphere_copy.get_state()))
        self.memory_queue.append([state,rewards])
        self.history.append([state,rewards])#rewards should be a list of rewards for each action
        # return [np.argmax(lgts) for lgts in logits]
        return np.argmax(logits)
    
    # def train(self, states, actions, rewards, epochs=10):
    def train(self, states, rewards, epochs=10):
        states = [self._reformat_state(state=state) for state in states]
        states = np.array(states).reshape((-1, 6, 9))
        actions = np.array(actions)
        rewards = np.array(rewards)
        self.model.fit(states, rewards, epochs=epochs)
        self.save_model(self.model_save_path)
    def train_with_history(self):
        if len(os.listdir(self.history_folder))<5:
            return
        self.training_chunks=100
        training_data = []
        for _ in range(self.training_chunks):
            for run in self.history:
                random_idx = random.randint(self.model_sequence_length,len(run)-self.model_sequence_length)
                training_data.append(run[(random_idx-self.model_sequence_length):(random_idx+self.model_sequence_length)])
        states = [x[0] for x in training_data]
        rewards = [x[1] for x in training_data]
        self.train(states=states,rewards=rewards,epochs=self.epochs)
    def get_reward_from_state(self, state):
        return self.sphere.get_reward(state)

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = models.load_model(filepath)

    def _reformat_state(self,state):
        return  np.array([[self.color_to_int[row] for row in s] for s in state])



