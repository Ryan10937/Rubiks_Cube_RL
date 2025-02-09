import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class RubiksCubeSolver:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(6, 9)))#keep faces together
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(16))  # Output layer with 16 logits
        model.compile(optimizer='adam', loss='mse')
        return model

    def inference(self, state):
        state = np.array(state).reshape((1, 6, 9))
        logits = self.model.predict(state)
        return [np.argmax(lgts) for lgts in logits]
    def train(self, states, actions, rewards, epochs=10):
        states = np.array(states).reshape((-1, 2, 2))
        actions = np.array(actions)
        rewards = np.array(rewards)
        self.model.fit(states, rewards, epochs=epochs)

    def get_reward_from_state(self, state):
        # Placeholder for reward calculation logic
        return np.random.random()

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = models.load_model(filepath)



