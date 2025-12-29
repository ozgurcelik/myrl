"""Initial DQN implementation with target network and replay buffer"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import argparse
from collections import deque

# Hyperparameters
lr = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
capacity = 10000
batch_size = 256

# First, define the network for DQN
# This model will return the Q-values for actions given a state
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_count, output_dim):
        super(Net, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layer_count-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQN():
    def __init__(self, state_size, action_size, lr, gamma, epsilon, epsilon_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.model = Net(state_size, 128, 2, action_size)
        self.target_model = Net(state_size, 128, 2, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=capacity)

    def act(self, state):
        """Returns an action based on the current state"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()
    
    def reward_function(self, original_reward, observation):
        """Returns a reward based on the observation"""
        return original_reward - 0.2 * np.abs(observation[0])
    
    def store(self, state, action, reward, next_state, done):
        """Stores the transition in the replay buffer"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([[action]], dtype=torch.long)
        reward = torch.tensor([[reward]], dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        done = torch.tensor([[done]], dtype=torch.float32)
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def learn(self):
        """Updates the model based on the observed transition"""
        # Sample a random batch of transitions from the replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        
        for state, action, reward, next_state, done in batch:
            current_q = self.model(state).gather(1, action)
            with torch.no_grad():
                next_q = self.target_model(next_state).max(dim=1)[0]
                target_q = reward + self.gamma * next_q * (1 - done)
            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.target_model.load_state_dict(self.model.state_dict())
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        

        # state, action, reward, next_state, done = random.sample(self.replay_buffer, batch_size)

        # # Get Q-values for current state
        # current_q = self.model(state).gather(1, action)

        # # Calculate the target Q-value
        # with torch.no_grad():
        #     next_q = self.model(next_state).max(dim=1)[0]
        #     target_q = reward + self.gamma * next_q * (1 - done)

        # # Calculate the loss
        # loss = self.loss_fn(current_q, target_q)

        # # Update the model
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # # Update the target model
        # self.target_model.load_state_dict(self.model.state_dict())

        # # Update epsilon
        # self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
    
def main():

    # Take command line arguments for environment
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    args = parser.parse_args()

    env = gym.make(args.env)
    observation, info = env.reset()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    dqn = DQN(state_size, action_size, lr, gamma, epsilon, epsilon_decay)
    rewards = []
    for episode in range(1000):
        state, info = env.reset()
        cum_reward = 0
        for t in range(1000):
            action = dqn.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            reward = dqn.reward_function(reward, next_state)
            dqn.store(state, action, reward, next_state, terminated or truncated)
            state = next_state
            cum_reward += reward
            if terminated or truncated:
                if len(dqn.replay_buffer) >= batch_size:
                    dqn.learn()
                break
        rewards.append(cum_reward)
        print(f"Episode {episode} finished after {t} steps with reward {cum_reward:.2f}")
    # Check if the results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")
    # Save the model
    torch.save(dqn.model.state_dict(), f"results/{args.env}_dqn.pth")
    # Create a plot of the rewards
    # also, add a line for the average reward over the last 10 episodes
    plt.plot(rewards)
    plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'))
    plt.title("Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(f"results/{args.env}_dqn.png")
    plt.close()
    env.close()

if __name__ == "__main__":
    main()
