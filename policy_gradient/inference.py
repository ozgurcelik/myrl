"""Inference script for the policy gradient model"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import argparse
import torch.distributions as ptd

def np2torch(x):
    return torch.tensor(x, dtype=torch.float32)

class Network(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, hidden_dim):
        super(Network, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class Policy(nn.Module):
    def __init__(self, network, action_dim):
        super(Policy, self).__init__()
        self.network = network
        self.action_dim = action_dim
        self.std = torch.zeros(action_dim) + 1e-6

    def action_distribution(self, observations):
        mean = self.network(observations)
        distribution = ptd.MultivariateNormal(loc=mean, covariance_matrix=torch.diag(self.std))
        return distribution
    
    def act(self, observations):
        observations = np2torch(observations)
        action_distribution = self.action_distribution(observations)
        sampled_actions = action_distribution.sample()
        sampled_actions = sampled_actions.detach().numpy()
        return sampled_actions
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--model", type=str, default="policy_gradient")
    args = parser.parse_args()
    
    env = gym.make(args.env, render_mode="human")
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    network = Network(observation_dim, action_dim, n_layers=2, hidden_dim=64)
    policy = Policy(network, action_dim)
    policy.network.load_state_dict(torch.load(f"results/{args.env}_{args.model}.pth"))
    
    state, info = env.reset()
    cum_reward = 0
    while True:
        action = policy.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
        if terminated or truncated:
            break
    print(f"Cumulative reward: {cum_reward}")
    env.close()

if __name__ == "__main__":
    main()