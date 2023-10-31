""" Performs value estimations for actions same as the incremental method.
The catch here is the distribution is not stationary meaning that the parameter of the distribution may not be same as previous step.
The max 
"""
import math
from typing import *

import numpy as np
import matplotlib.pyplot as plt

from utils import pull_arm, choose_optimally, create_reward_distribution, update_distribution

    
def run_experiement(k_distribution, steps: int, initial_value: float,  epsilon: float) -> Tuple[List[float], List[float]]:
    """Runs one full experitments with given number of steps
    Args:
        k_distribution: probability distribution for each action
        steps: number times choices to be made.
        initial_value: initial value of all actions
        epsilon: parameter for epsilon greedy choice
    Returns:
        rewards: list of reward obtained from each step
        values_of_action: values of each action calculated by action value method
    """  


    k = len(k_distribution)
    rewards = []
    # values_of_actions = [0] * k
    # Optimal Initialization
    values_of_actions = [initial_value] * k
    n_of_actions = [0] * k

    max_rewards = []

    alpha = lambda x: math.sin(n_of_actions[x])**2
    for _ in range(steps):
        choice = choose_optimally(values_of_actions, epsilon)
        reward = pull_arm(choice, k_distribution) 
        rewards.append(reward)

        n_of_actions[choice] += 1
        values_of_actions[choice] += (reward - values_of_actions[choice])* alpha(choice)

        max_rewards.append(max(map(lambda x: x[0] , k_distribution)))

        k_distribution = update_distribution(k_distribution, 0.01, 0.03)

    return rewards, values_of_actions, max_rewards


if __name__ == "__main__":
    K = 20
    k_distribution = create_reward_distribution(K)


    experiment_configs = [
        (5000, 0,  0.01),
        (5000, 0, 0.1),
        (5000, 5,  0.01),
        (5000, 5, 0.1),
    ]

    for config in experiment_configs:
        rs, vs, mxrs = run_experiement(k_distribution, *config)

        cum_sum = 100 * np.cumsum(rs) / np.cumsum(mxrs)
        plt.plot(cum_sum, label=f"epsilon - {config[2]} initial-val - {config[1]}")
    plt.xscale('log')
    plt.title('Reward')
    plt.legend()
    plt.xlabel('Choice')
    plt.ylabel('% of Optimal Action')
    plt.ylim([0, 100])
    plt.show()
