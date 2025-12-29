# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("HalfCheetah-v5", render_mode="human")

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# info: extra debugging information (usually not needed for basic learning)

print(f"Starting observation: {observation}")
# Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()

# # import gymnasium as gym
# # env = gym.make("CartPole-v1")
# # observation, info = env.reset()
# # print(f"Starting observation: {observation}")
# # print(f"Info: {info}")
# # print(f"Action space: {env.action_space}")
# # print(f"Observation space: {env.observation_space}")
# # print(f"State size: {env.observation_space.shape[0]}")
# # print(f"Action size: {env.action_space.n}")

# import numpy as np
# from collections import deque
# from random import sample

# # array = np.array([1, 2, 3, 4, 5])
# # deq = deque(array)
# # print(deq)
# # random_sample = sample(deq, 2, )
# # print(random_sample)
# # print(deq)
# # a2 = list(deq)
# # print(a2)

# import gymnasium as gym
# env = gym.make("HalfCheetah-v5")
# observation, info = env.reset()
# print(f"Starting observation: {observation}")
# print(f"Info: {info}")
# print(f"Action space: {env.action_space}")
# print(f"Observation space: {env.observation_space}")
# print(f"State size: {env.observation_space.shape[0]}")
# print(f"Action size: {env.action_space.shape[0]}")
