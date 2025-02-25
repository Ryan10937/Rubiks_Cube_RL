from difflib import restore
from turtle import st
from colorama import init
from matplotlib.pylab import rand
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LSTM, Dense, Reshape, TimeDistributed, Masking
import numpy as np
import random
import os
from sphere import RubiksCube
import copy
import csv

class RubiksCubeSolver:
    def __init__(self,show_plot=True,eps=0.1,episode=0):
        self.sphere = RubiksCube(show_plot=show_plot)
        self.epsilon_decay = 0.995 ** (episode+1)# Exploration rate decay
        self.epsilon = eps * self.epsilon_decay# Exploration rate
        self.memory_queue = [] # Queue of states
        self.history = [] #array of states and rewards
        self.history_path = 'history'
        self.model_save_path = 'models/lstm_5_seq_masking_long.keras'
        self.action_range = 12
        self.model_sequence_length = 10 #prev, 5
        self.epochs = 100
        self.batch_size = 2**10
        self.training_chunks = 100_000
        self.model = self._build_model()
        self.color_to_int={
            'green':0,
            'purple':1,
            'red':2,
            'blue':3,
            'yellow':4,
            'orange':5,
            'null':-1,
        }

    def save_history(self):
        filepath = f'{self.history_path}/history_{len(os.listdir(self.history_path))+1}.npy'
        with open(filepath, 'a', newline='') as csvfile:
            for hist,reward in self.history:
                writer = csv.writer(csvfile)
                writer.writerow([hist.tolist(), reward])
    
    def load_history(self):
        history = []
        for file in os.listdir(self.history_path):
            run_history = []
            with open(os.path.join(self.history_path, file), 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    hist = np.array(eval(row[0]))
                    reward = [float(r) for r in row[1].strip('[]').split(',')]
                    if len(hist) == 0:
                        print('Empty history')
                        continue
                    if len(reward) == 0:
                        print('Empty history')
                        continue
                    run_history.append([hist, reward])
                history.append(run_history)
        return history
        
    def _build_model(self):
        if os.path.exists(self.model_save_path):
            model = models.load_model(self.model_save_path)
            return model
        else:    
            # Define the model
            model = models.Sequential()
            # Reshape input to (n, s, 54)
            model.add(TimeDistributed(Reshape((54,)), input_shape=(self.model_sequence_length, 6, 9)))
            # Add Masking layer
            # model.add(Masking(mask_value=-1, input_shape=(self.model_sequence_length, 6, 9)))
            model.add(Masking(mask_value=-1))
            # Add LSTM layer(s)
            model.add(LSTM(64, return_sequences=True))  
            model.add(LSTM(64, return_sequences=True))  
            model.add(LSTM(64, return_sequences=False))  
            # Add a Dense layer to map to 12 outputs
            model.add(Dense(12))
            # Compile the model
            model.compile(optimizer='adam', loss='mse')  # Adjust loss and optimizer as needed
            return model


    def infer(self, state):
        state = self._reformat_state(states=[state])
        state = np.array(state).reshape((1, 6, 9))

        if random.random() < self.epsilon:
            action = random.randint(0, self.action_range)
        else:
            if len(self.memory_queue) <= self.model_sequence_length:
                action = random.randint(0, self.action_range)
            else:
                memory_states = np.array([x[0] for x in self.memory_queue])
                state_with_memory = np.concatenate((memory_states, state), axis=0)
                action = np.argmax(self.model.predict(np.array([state_with_memory]),verbose=0))
        #add to memory queue
        while len(self.memory_queue) > self.model_sequence_length:
            self.memory_queue.pop(0)
        rewards = self.sphere.get_next_state_rewards()
        self.memory_queue.append([state[0],rewards])
        self.history.append([state[0],rewards])#rewards should be a list of rewards for each action
        return action
    
    def train(self, states, rewards, epochs=10):
        print('Training model')
        #this function expects a list of states in color format
            #eg [['green','green','green',...],['purple','purple','purple',...],...]
            #and they must be instances of 6 by 9
        states = [self._reformat_state(states=state) for state in states]
        states = np.array(states).reshape((-1,self.model_sequence_length,6, 9))
        rewards = np.array([r[-1] for r in rewards])
        self.model.fit(states, 
                       rewards, 
                       epochs=epochs,
                       validation_split=0.2,
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)],
                       batch_size = self.batch_size,
                       )
        self.save_model(self.model_save_path)
    
    def train_with_history(self):
        if len(os.listdir(self.history_path))<1:
            return
        history = self.load_history()
        for hist in history:
            if len(hist)==0:
                print('Empty entry')
                continue
            for hist_reward_pair in hist:
                if len(hist_reward_pair[0])==0:
                    print('Empty history')
                if len(hist_reward_pair[1])==0:
                    print('Empty reward')

        training_data = []
        for _ in range(self.training_chunks):
            hist = history[random.randint(0,len(history)-1)]#should pick one file
            if len(hist)<self.model_sequence_length:
                continue
                fill_char = -1
                hist_to_append = []
                for _ in range(self.model_sequence_length-len(hist)):
                    hist_to_append.append([np.array([[fill_char for _ in range(9)] for x in range(6)]),
                                           [0 for x in range(12)]])
                for h in hist:
                    hist_to_append.append(h)
            else:
                random_idx = random.randint(self.model_sequence_length,len(hist))
                hist_to_append = hist[(random_idx-self.model_sequence_length):random_idx]
                if len(hist_to_append) == 0:
                    print('Empty history in train')
                    continue
            training_data.append(hist_to_append)

        #training data is n sets of self.model_sequence length state,reward pairs
        states = [[y[0] for y in x] for x in training_data]
        rewards = [[y[1] for y in x] for x in training_data]
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

    def _reformat_state(self,states):
        # if type(states[0][0][0]) == np.int64:
        if type(states[0][0][0]) == np.int64:
            return states
        return  np.array([[[self.color_to_int[row] for row in s] for s in state] for state in states])



