import atexit
import pickle
import os
from collections import deque
import random as rd

import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from ple import PLE
from ple.games import FlappyBird
import numpy as np


def parse_game_state(state):
    state_vector = []
    state_vector.append(state.get('player_y'))
    state_vector.append(state.get('player_vel'))
    state_vector.append(state.get('next_pipe_dist_to_player'))
    state_vector.append(state.get('next_pipe_top_y', ))
    state_vector.append(state.get('next_pipe_bottom_y'))
    state_vector.append(state.get('next_next_pipe_dist_to_player'))
    state_vector.append(state.get('next_next_pipe_top_y'))
    state_vector.append(state.get('next_next_pipe_bottom_y'))
    return np.reshape(np.matrix(state_vector), None, 8)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(8, input_dim=8, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return rd.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = rd.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    game = FlappyBird()
    game.allowed_fps = 30
    p = PLE(game, fps=30, display_screen=True)
    agent = DQNAgent(8, 2)
    if os.path.exists('memory.h5py'):
        agent.load('memory.h5py')
    atexit.register(agent.save, 'memory.h5py')

    p.init()
    reward = 0.0
    max_score = 0
    current_score = 0
    while True:
        state = parse_game_state(game.getGameState())
        for frame in range(1000):
            action = np.argmax(agent.model.predict(state)[0])
            p.act(119 if action == 1 else None)

            next_state = parse_game_state(game.getGameState())
            reward = p.act(119 if action == 1 else None)

            done = False
            if not game.game_over():
                reward = abs(next_state[0, 0] - next_state[0, 3] + 50) * 0.05
                if game.getScore() > current_score:
                    reward = 1
                    current_score = game.getScore()
            else:
                current_score = 0
                reward = abs(next_state[0, 0] - next_state[0, 3] + 50) * (-0.05)
                done = True
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                p.reset_game()
                break
            score = game.getScore()
            if score > max_score:
                max_score = score
                print("=" * 40)
                print(max_score)
                print("=" * 40)
            time.sleep(0.03)

        if len(agent.memory) > 32:
            agent.replay(32)

