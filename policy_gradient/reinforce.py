"""REINFORCE"""

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
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def action_distribution(self, observations):
        mean = self.network(observations)
        std = torch.exp(self.log_std)
        distribution = ptd.MultivariateNormal(loc=mean, covariance_matrix=torch.diag(std))
        return distribution
    
    def act(self, observations):
        observations = np2torch(observations)
        action_distribution = self.action_distribution(observations)
        sampled_actions = action_distribution.sample()
        sampled_actions = sampled_actions.detach().numpy()
        return sampled_actions


class PolicyGradient(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.network = Network(self.observation_dim, self.action_dim, n_layers=2, hidden_dim=64)
        self.baseline = Network(self.observation_dim, 1, n_layers=2, hidden_dim=64)
        self.policy = Policy(self.network, self.action_dim)
        self.num_batches = 200  # number of batches trained on
        self.batch_size = 10000  # number of steps used to compute each policy update
        self.max_ep_len = 1000  # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 0.9  # the discount factor
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=self.learning_rate)


    def sample_path(self):
        step_count = 0
        paths = []

        while step_count < self.batch_size:
            state, info = self.env.reset()
            states, actions, rewards = [], [], []

            t = 0
            while t < self.max_ep_len:
                states.append(state)
                # [None] adds a batch dimension to the state. 
                # So, if the state was a vector of shape (17,), it becomes (1, 17). 
                # The network expects a batch of inputs, and this makes the single state look like a batch of size one.
                action = self.policy.act(states[-1].reshape(1, -1))[0]
                state, reward, terminated, truncated, info = self.env.step(action)
                actions.append(action)
                rewards.append(reward)
                t += 1
                if terminated or truncated:
                    break
            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
            }
            paths.append(path)
            step_count += t
        return paths
    
    def get_returns(self, paths):
        path_returns = []
        for path in paths:
            returns = np.zeros_like(path['reward'])
            returns[-1] = path['reward'][-1]
            for i in range(len(returns)-2, -1, -1):
                returns[i] = path['reward'][i] + self.gamma * returns[i+1]
            path_returns.append(returns)
        path_returns = np.array(path_returns)
        return path_returns
    
    def calculate_advantage(self, targets, observations):
        """Advantage: A_t = target_t - V(s_t)"""
        observations = np2torch(observations)
        baseline = self.baseline(observations).detach().numpy()
        baseline = baseline.squeeze()
        advantages = targets - baseline
        return advantages
    
    def get_td_targets(self, paths):
        """
        Compute TD(0) targets: r_t + gamma * V(s_{t+1})
        For terminal states, the target is just r_t (no bootstrapping)
        """
        all_td_targets = []
        for path in paths:
            observations = path['observation']
            rewards = path['reward']
            T = len(rewards)
            
            # Get value estimates for all states
            obs_tensor = np2torch(observations)
            values = self.baseline(obs_tensor).detach().numpy().squeeze()
            
            td_targets = np.zeros(T)
            for t in range(T):
                if t == T - 1:
                    # Terminal state: target is just the reward
                    td_targets[t] = rewards[t]
                else:
                    # TD(0) target: r_t + gamma * V(s_{t+1})
                    td_targets[t] = rewards[t] + self.gamma * values[t + 1]
            
            all_td_targets.append(td_targets)
        return all_td_targets
    
    def normalize_advantage(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def update_baseline(self, targets, observations):
        """Update baseline to predict targets"""
        observations = np2torch(observations)
        targets = np2torch(targets)
        baseline = self.baseline(observations).squeeze()
        loss = nn.functional.mse_loss(baseline, targets)
        self.baseline_optimizer.zero_grad()
        loss.backward()
        self.baseline_optimizer.step()
    
    def update_policy(self, observations, actions, advantages):
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        
        self.optimizer.zero_grad()
        dist = self.policy.action_distribution(observations)
        log_probs = dist.log_prob(actions)
        loss = -torch.sum(log_probs*advantages)/observations.shape[0]
        loss.backward()
        self.optimizer.step()

    def train(self, method='monte_carlo'):
        """
        Train the policy using different advantage estimation methods.
        
        Args:
            method: 'monte_carlo' or 'td'
        """
        avg_rewards = []
        for i in range(self.num_batches):
            paths = self.sample_path()
            
            observations = np.concatenate([path['observation'] for path in paths])
            actions = np.concatenate([path['action'] for path in paths])
            rewards = np.concatenate([path['reward'] for path in paths])
            avg_rewards.append(np.array([path['reward'].sum() for path in paths]).mean())

            if method == 'monte_carlo':
                # Monte Carlo approach: use full episode returns
                targets = self.get_returns(paths)
                targets = np.concatenate(targets)
            elif method == 'td':
                # TD(0) approach: use bootstrapped targets
                targets = self.get_td_targets(paths)
                targets = np.concatenate(targets)
            else:
                raise ValueError(f"Unknown method: {method}. Choose from 'monte_carlo' or 'td'")
            
            advantages = self.calculate_advantage(targets, observations)
            advantages = self.normalize_advantage(advantages)
            self.update_baseline(targets, observations)
            self.update_policy(observations, actions, advantages)

            if (i+1) % 10 == 0:
                print(f"Batch {i} average reward: {avg_rewards[-1]}")

        return avg_rewards
            
    
def main():
    # Take command line arguments for environment
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    args = parser.parse_args()

    pg = PolicyGradient(args.env)
    avg_rewards = pg.train()
    # Check if the results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")
    # Save the model
    torch.save(pg.policy.network.state_dict(), f"results/{args.env}_reinforce.pth")
    # Create a plot of the rewards
    # also, add a line for the average reward over the last 10 episodes
    plt.plot(avg_rewards)
    plt.plot(np.convolve(avg_rewards, np.ones(10)/10, mode='valid'))
    plt.title("Rewards")
    plt.xlabel("Batch")
    plt.ylabel("Average Reward")
    plt.savefig(f"results/{args.env}_reinforce.png")
    plt.close()

if __name__ == "__main__":
    main()
