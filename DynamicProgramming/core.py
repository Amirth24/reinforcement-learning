"""
    Contains all required and important functions to obtain an 
    optimal policy through dynamic program
"""
import gymnasium
import numpy as np


def compute_optimal_value_with_vi(
    env: gymnasium.Env,
    gamma: float = 0.5,
    max_iter: int = 10000,
    threshold: float = 1e-10,
) -> np.array:
    """Calculate the state value for the given environment
    Args:
        env (Env): The Gymnasium Environment
        gamma (float): The discount factor (default=0.0)
        threshold (float): Theta value for checking convergence
    Returns:
        np.array: Value of the states"""
    value_table = np.zeros(env.observation_space.n)

    for i in range(max_iter):
        updated_valute_table = np.copy(value_table)

        for state in range(env.observation_space.n):
            q_val = []
            for action in range(env.action_space.n):
                next_state_rewards = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_state_rewards.append(
                        trans_prob
                        * (reward_prob + gamma * updated_valute_table[next_state])
                    )

                q_val.append(sum(next_state_rewards))
            value_table[state] = max(q_val)

        if np.sum(np.fabs(updated_valute_table - value_table)) <= threshold:
            print("Value-iteration Converged at iteration ", i + 1)
            break

    return value_table


def compute_value_function(
    env: gymnasium.Env,
    policy: np.array,
    gamma: float = 0.5,
    threshold: float = 1e-10,
) -> np.array:
    """Calculate the state value for the given policy
    Args:
        env (Env): The Gymnasium Environment
        policy (np.array): The action distribution for all state.
        gamma (float): The discount factor (default=0.0)
        threshold (float): Theta value for checking convergence
    Returns:
        np.array: Value of the states
    """
    value_table = np.zeros(env.observation_space.n)

    i = 0
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            action = policy[state]
            value_table[state] = sum(
                [
                    trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                    for trans_prob, next_state, reward_prob, _ in env.unwrapped.P[
                        state
                    ][action]
                ]
            )

        i += 1
        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            print("Value converged at", i)
            break

    return value_table


def extract_policy(
    env: gymnasium.Env, value_table: np.array, gamma: float = 0.5
) -> np.array:

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
                q_table[action] += trans_prob * (
                    reward_prob + gamma * value_table[next_state]
                )

        policy[state] = np.argmax(q_table)
    return np.int8(policy)


def optimal_policy_with_pi(
    env: gymnasium.Env, gamma: float = 0.5, max_iter: int = 20000
) -> np.array:
    """Finds the optimal policy for the given environement
    Args:
        env (Env): The  Gymnasium Environment.
        gamma (float): The discount factor (default=0.5)
        max_iter (int) : The maximum no of iteration.
    Returns
        np.array : The optimal policy
    """
    policy = np.zeros(env.observation_space.n)
    for j in range(max_iter):
        value = compute_value_function(env, policy, gamma)

        new_policy = extract_policy(env, value, gamma)

        if np.all(policy == new_policy):
            print("Policy is stablized at", j + 1)
            break

        policy = new_policy.astype(int)
  