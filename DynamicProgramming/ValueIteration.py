"""
    Value Iteration

    In value iteration,the value of states randomly intialized. 
    Then iteratively for each state new value function is found.
    Until a optimal value function is found this is repeated. Once
    optimal value function is found policy can be extracted from it.
"""

import gymnasium as gym

from core import compute_optimal_value_with_vi, extract_policy

# Creating the environment
# Changing is_slippery false make the actions deterministic
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
env.reset()


observation = 0
gamma = 0.8

optimal_value_function = compute_optimal_value_with_vi(env, gamma)

optimal_policy = extract_policy(env, optimal_value_function, gamma)


print("Optimal Policy", optimal_policy)
total_reward = 0
for i in range(1000):
    observation, reward, term, trunc, _ = env.step(optimal_policy[observation])
    total_reward += reward
    if trunc or term:
        print("Average Reward: ", total_reward / i)
        break

    env.render()

env.close()
