""" 
Performs value estimations for actions in multi arm bandit problem with K action
The update rule used here is same as used in the incremental method in incremental 
implementation of action value method
"""
from typing import *

import numpy as np
import matplotlib.pyplot as plt

from utils import pull_arm, choose_optimally, create_reward_distribution


def run_experiments(
    k_distribution, steps: int, epsilon: float
) -> Tuple[List[float], List[float]]:
    """Runs one full experitments with given number of steps
    Args:
        k_distribution: probability distribution for each action
        steps: number times choices to be made.
        epsilon: parameter for epsilon greedy choice
    Returns:
        rewards: list of reward obtained from each step
        values_of_action: values of each action calculated by action value method
    """
    k = len(k_distribution)
    rewards = []
    values_of_actions = [0] * k
    n_of_actions = [0] * k
    max_rewards = []
    for _ in range(steps):
        choice = choose_optimally(values_of_actions, epsilon)
        reward = pull_arm(choice, k_distribution)
        rewards.append(reward)
        n_of_actions[choice] += 1
        values_of_actions[choice] += (reward - values_of_actions[choice]) * (
            1 / n_of_actions[choice]
        )

        max_rewards.append(max(map(lambda x: x[0], k_distribution)))

    return rewards, values_of_actions, max_rewards


if __name__ == "__main__":
    K = 10
    k_distribution = create_reward_distribution(K)

    experiment_configs = [
        (5000, 0.01),
        (5000, 0.1),
        (5000, 0.3),
        (5000, 0),
        (5000, 1),
        (5000, 0.5),
    ]

    for config in experiment_configs:
        rs, vs, mxrs = run_experiments(k_distribution, *config)

        cum_sum = 100 * np.cumsum(rs) / np.cumsum(mxrs)
        plt.plot(cum_sum, label=f"epsilon - {config[1]}")
    plt.xscale("log")
    plt.title("Reward")
    plt.legend()
    plt.xlabel("Choice")
    plt.ylabel("% of Optimal Action")
    plt.ylim([0, 100])
    plt.show()
