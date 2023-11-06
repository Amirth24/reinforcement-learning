"""
    Policy Iteration
    In policy iteration a random policy is initilaized first, then 
    it will be evaluated by computing the value functions for them. 
    If it is not a good policy find a better one. Repeat the process
    until the old policy is same as the new one.
"""
import pygame
import gymnasium as gym
import numpy as np

from utils import compute_state_value, extract_policy

# Creating the environment
# Changing is_slippery false make the actions deterministic
env = gym.make(
    "FrozenLake-v1", 
    is_slippery=False, 
    render_mode='human'
)
env.reset()

# Creating a random policy
policy = np.random.randint(0, 4, env.observation_space.n)

print(policy)
observation = 0
gamma = 0.8
for i in range(20000):
    value = compute_state_value(env, policy, gamma)

    new_policy = extract_policy(env, value, gamma)

    if np.all(policy == new_policy):
        print("Policy is stablized at", i + 1)
        break

    policy  = new_policy.astype(int)

print("Optimal Policy", policy)

for i in range(1000):
    observation, reward, term, trunc, _ = env.step(
        policy[observation]
    )

    if trunc or term:
        print("State Restarted")
        env.reset()

    env.render()

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            pygame.quit()

env.close()
