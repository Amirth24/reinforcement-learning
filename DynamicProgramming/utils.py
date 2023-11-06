"""
    Contains all required and important functions to obtain an 
    optimal policy through dynamic program
"""
import gymnasium
import numpy as np


def compute_state_value(
    env: gymnasium.Env,
    policy: np.array,
    gamma: float = 0.5,
    threshold: float = 1e-10,
) -> np.array:
    """Calculate the state value for the give policy
    Args:
        env (Env): The Gymnasium Environment
        policy (np.array): The action distribution for all state.
        gamma (float): The discount factor (default=0.0)
        threshold (float): Theta value for checking convergence
    Returns:
        np.array: Value of the states
    """
    value_table = np.zeros(env.observation_space.n)

    # for i in range(max_iter):
    i= 0

    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            action = policy[state]
            value_table[state] = sum([
                trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                for trans_prob, next_state, reward_prob, _ in env.unwrapped.P[state][action]
            ])

        i += 1
        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            print("Value converged at", i)
            break

    return value_table



def extract_policy(env: gymnasium.Env, value_table: np.array, gamma:float = 0.5) -> np.array:
    """Extract a better policy for the given value table
    Args:
        env (Env): The Gymnasium Environment
        value_table (np.array): The computed value of states.
        gamma (float): The discount factor (default=0.5)
    Returns:
        np.array: Optimal Policy for the given value function

    """
    policy = np.zeros(env.observation_space.n)

    for state in range(env.observation_space.n):
        q_table = np.zeros(env.action_space.n)

        for action in range(env.action_space.n):
            for next_sr in env.unwrapped.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                q_table[action] += trans_prob * (reward_prob + gamma * value_table[next_state])

        policy[state] = np.argmax(q_table)
    return policy


                q_values[state, action] = np.sum(action_values)
        
        if np.sum(np.fabs(updated_q_values - q_values)) < threshold:
            print('Value converged at', i+1)

            break
        
    return q_values
