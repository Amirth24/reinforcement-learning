import gymnasium
import numpy as np


def compute_state_value(
    env: gymnasium.Env,
    policy: np.array,
    gamma: float = 0.0,
    max_iter: int = 100,
    threshold: float = 1e-20,
) -> np.array:
    """Calculate the state value for the give policy
    Args:
        env (Env): The The Gymnasium Environment
        policy (np.array): The action distribution for all state.
        gamma (float): The discount factor (default=0.0)
        max_iter (int): Maximum no of iterations
        threshold (float): Theta value for checking convergence
    Returns:
        np.array: Value of the states
    """
    value_table = np.zeros(env.observation_space.n)

    for i in range(max_iter):
        updated_value_table = np.copy(value_table)
        for st in range(env.observation_space.n):
            val = []
            for action in range(env.action_space.n):
                next_state_reward = []
                for next_sr in env.unwrapped.P[st][action]:
                    (
                        trans_prob,
                        next_state,
                        reward_prob,
                        _,
                    ) = next_sr
                    next_state_reward.append(
                        trans_prob
                        * (reward_prob + gamma * updated_value_table[next_state])
                    )

                val.append(policy[st, action] * np.sum(next_state_reward))

            value_table[st] = np.sum(val)

        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            print("Value converged at ", i + 1)
            break

    return value_table

def compute_state_action_value(
    env: gymnasium.Env,
    gamma: float = 0.0,
    max_iter: int = 100,
    threshold: float = 1e-20,
):
    """Calculate the state action value for the give policy
    Args:
        env (Env): The The Gymnasium Environment
        policy (np.array): The action distribution for all state.
        gamma (float): The discount factor (default=0.0)
        max_iter (int): Maximum no of iterations
        threshold (float): Theta value for checking convergence
    Returns:
        np.array: Value of the states action pair
    """
    q_values = np.zeros(
        (env.observation_space.n, env.action_space.n)
    )

    for i in range(max_iter):
        updated_q_values = np.copy(q_values)
        
        for state in range(env.observation_space.n):

            for action in range(env.action_space.n):
                action_values = []
                for next_sr in env.get_wrapper_attr('P')[state][action]:
                    (
                        trans_prob,
                        next_state,
                        reward,
                        _
                    ) = next_sr 
                    action_values.append(trans_prob * (reward + gamma * np.max(q_values[next_state])))
                

                q_values[state, action] = np.sum(action_values)
        
        if np.sum(np.fabs(updated_q_values - q_values)) < threshold:
            print('Value converged at', i+1)

            break
        
    return q_values
