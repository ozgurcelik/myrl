"""This script is used to load a trained DQN model and run it in the environment"""

import torch
import torch.nn as nn
import gymnasium as gym
import argparse

# Define the network architecture, same as in basic_dqn.py
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_count, output_dim):
        super(Net, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layer_count - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
def main():

    # Take command line arguments for environment
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--model", type=str, default="basic_dqn")
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="human")
    observation, info = env.reset()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Instantiate the model with the same architecture and parameters as in training
    model = Net(state_size, 128, 2, action_size)

    # Load the saved weights into the model instance
    model.load_state_dict(torch.load(f"results/{args.env}_{args.model}.pth"))
    model.eval()

    def act(state):
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32)).unsqueeze(0)
            return torch.argmax(q_values, dim=1).item()

    t = 0
    while True:
        action = act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        t += 1
        if terminated or truncated:
            print(f"Episode finished after {t} steps")
            break

    env.close()

if __name__ == "__main__":
    main()