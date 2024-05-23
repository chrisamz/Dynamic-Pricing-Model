# src/reinforcement_learning.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define file paths
processed_data_path = 'data/processed/processed_data.csv'
rl_model_path = 'models/rl_pricing_model.h5'
results_path = 'results/rl_pricing_results.txt'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(os.path.dirname(rl_model_path), exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# Load processed data
print("Loading processed data...")
data = pd.read_csv(processed_data_path)

# Ensure date column is in datetime format
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

# Set date as index
data.set_index('date', inplace=True)

# Define the state and action space
states = data[['sales', 'price', 'rolling_mean_7', 'rolling_std_7']].values
actions = np.linspace(data['price'].min(), data['price'].max(), 10)

# Normalize the states
scaler = MinMaxScaler()
states = scaler.fit_transform(states)

# Define the DQN model
def build_dqn_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(24, input_shape=input_shape, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_dqn_model((self.state_size,), self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Initialize the DQN agent
state_size = states.shape[1]
action_size = len(actions)
agent = DQNAgent(state_size, action_size)

# Train the DQN agent
episodes = 1000
batch_size = 32

for e in range(episodes):
    state = states[0].reshape(1, -1)
    for time in range(1, len(states)):
        action_idx = agent.act(state)
        action = actions[action_idx]
        next_state = states[time].reshape(1, -1)
        reward = next_state[0][0]  # Reward is the sales value
        done = time == len(states) - 1
        agent.remember(state, action_idx, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e+1}/{episodes}, Time: {time}, Epsilon: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

# Save the trained model
agent.save(rl_model_path)

# Evaluate the DQN agent
print("Evaluating the DQN agent...")
total_reward = 0
state = states[0].reshape(1, -1)
rewards = []

for time in range(1, len(states)):
    action_idx = agent.act(state)
    action = actions[action_idx]
    next_state = states[time].reshape(1, -1)
    reward = next_state[0][0]
    total_reward += reward
    rewards.append(reward)
    state = next_state

print(f"Total Reward: {total_reward:.2f}")

with open(results_path, 'w') as f:
    f.write(f"Total Reward: {total_reward:.2f}\n")

# Plot the rewards over time
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title('Rewards Over Time')
plt.xlabel('Time')
plt.ylabel('Reward')
plt.savefig(os.path.join(figures_path, 'rewards_over_time.png'))
plt.show()

print("Reinforcement learning completed!")
