from turtle import st
from colorama import init
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random
import os
from sphere import Sphere
import copy
import csv

class RubiksCubeSolver:
    def __init__(self):
        self.model = self._build_model()
        self.sphere = Sphere()
        self.epsilon = 0.2 # Exploration rate
        self.memory_queue = [] # Queue of states
        self.history = [] #array of states and rewards
        self.history_path = 'history'
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
        with open(filepath, 'a', newline='') as csvfile:
            for hist,reward in self.history:
                writer = csv.writer(csvfile)
                writer.writerow([hist.tolist(), reward])
    def load_history(self):
        history = []
        for file in os.listdir(self.history_path):
            with open(os.path.join(self.history_path, file), 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    hist = np.array(eval(row[0]))
                    reward = [float(r) for r in row[1].strip('[]').split(',')]
                    # history.append([hist, reward])
                    history.append([hist[0], reward])
        return history
                    
        
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
        state = self._reformat_state(state=state)
        state = np.array(state).reshape((1, 6, 9))

        if random.random() < self.epsilon:
            action = random.randint(0, self.action_range)
        else:
            if len(self.memory_queue) < self.model_sequence_length:
                action = random.randint(0, self.action_range)
            else:
                memory_states = np.array([x[0] for x in self.memory_queue])
                state_with_memory = np.concatenate((memory_states, state), axis=0)
                action = np.argmax(self.model.predict(state_with_memory,verbose=0))
        #add to memory queue
        while len(self.memory_queue) > self.model_sequence_length:
            self.memory_queue.pop(0)
        rewards=[]
        for next_action in range(self.action_range):
            sphere_copy = Sphere()
            sphere_copy.points = copy.deepcopy(self.sphere.points)
            sphere_copy.move(next_action)
            rewards.append(self.get_reward_from_state(sphere_copy.get_state()))
        self.memory_queue.append([state[0],rewards])
        self.history.append([state,rewards])#rewards should be a list of rewards for each action
        return action
    
    def train(self, states, rewards, epochs=10):
        print('Training model')
        #this function expects a list of states in color format
            #eg [['green','green','green',...],['purple','purple','purple',...],...]
            #and they must be instances of 6 by 9
        states = [self._reformat_state(state=state) for state in states]
        states = np.array(states).reshape((-1, 6, 9))
        rewards = np.array(rewards)
        self.model.fit(states, rewards, epochs=epochs)
        self.save_model(self.model_save_path)
    def train_with_history(self):
        if len(os.listdir(self.history_path))<1:
            return
        history = self.load_history()

        self.training_chunks=100
        training_data = []
        for _ in range(self.training_chunks):
            for hist in history:
                if len(hist[0])<self.model_sequence_length*2:
                    print('skipping run with length',len(hist[0]))
                    continue
                random_idx = random.randint(self.model_sequence_length,len(hist[0])-self.model_sequence_length)

                training_data.append(hist[(random_idx-self.model_sequence_length):(random_idx+self.model_sequence_length)])
        states = [x[0] for x in training_data]
        rewards = [x[1] for x in training_data]
        if len(training_data) == 0:
            print('No training data')
            return
        self.train(states=states,rewards=rewards,epochs=self.epochs)
    def get_reward_from_state(self, state):
        return self.sphere.get_reward(state)

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = models.load_model(filepath)

    def _reformat_state(self,state):
        # return  np.array([[self.color_to_int[row] for row in s] for s in state])
        print('state',state)
        if type(state[0][0]) == np.int64:
            return state
        return  np.array([[self.color_to_int[row] for row in s] for s in state])



