"""
Policy Evaluation (Prediction) using Dynamic Programming

Here the state-value function v_pi for an arbitary policy pi is computed. 
Consider v0, v1, v2 are mapping s+(state space with terminal state) to a 
value in (real number). The initial approximation v0 is chosen arbitarily
(except that the terminal state, if any, must be 0), and each successive 
approximation is obtained by using the Bellman equation. (refer Eq 4.5 page 74 
Bartto Suttan)

"""
import gymnasium as gym 
import numpy as np

# Creating the environment 
# Changing is_slippery false make the actions deterministic
env = gym.make(
    "FrozenLake-v1", 
    is_slippery=False, 
    render_mode='human'
)
env.reset()

# intialising the value table to zero
value_table = np.zeros(env.observation_space.n)
policy = np.full((env.observation_space.n, env.action_space.n), 1/env.action_space.n)
GAMMA = 0.3 # The discount factor
THRESHOLD = 1e-50 # Threshold theta for checking convergence


for i in range(1000):
    updated_value_table = np.copy(value_table)
    for st in range(env.observation_space.n):
        val = []

        for action in range(env.action_space.n):
            next_state_reward = []
            for next_sr in env.unwrapped.P[st][action]:
                trans_prob, next_state, reward_prob, _, = next_sr
                next_state_reward.append(
                    trans_prob * (reward_prob + GAMMA * updated_value_table[next_state])
                )

            val.append(policy[st, action] *np.sum(next_state_reward))

        value_table[st] = np.sum(val)

    if np.sum(np.fabs(updated_value_table - value_table)) <= THRESHOLD:
        print('Value converged at ', i + 1)
        break

print("Value of the policy: ")
print(value_table)

env.close()
