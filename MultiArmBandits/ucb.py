""" 
The idea of this upper confidence bound (UCB) action selection is that the square-root
term is a measure of the uncertainty or variance in the estimate of a's value. The quantity
being max'ed over is thus a sort of upper bound on the possible true value of action a, with
c determining the confidence level.
"""
import math
from typing import *

import numpy as np
import matplotlib.pyplot as plt

from utils import pull_arm, create_reward_distribution, update_distribution


def choose_arm_ucb(values: List[float], c: float, t: int, n_of_actions: List[int]):
    """ Choose an arm based on the UCB value 
    Args:
        values: value of actions
        c: controls the degree of exploration
        t: t-th step in the experiement
        n_of_actions: the number of occurences of each actions
    """
    values, n_of_actions = np.array(values), np.array(n_of_actions)
    return np.argmax(values + c * np.sqrt(math.log(t) / n_of_actions))


def run_experiement(k_distribution, steps: int, c: float) -> Tuple[List[float], List[float]]:
    """Runs one full experitments with given number of steps
    Args:
        k_distribution: probability distribution for each action
        steps: number times choices to be made.
        c: controls the degree of exploration
    Returns:
        rewards: list of reward obtained from each step
        values_of_action: values of each action calculated by action value method
    """

    k = len(k_distribution)
    rewards = []
    # Optimal Initialization
    values_of_actions = [0] * k
    n_of_actions = [1] * k

    max_rewards = []

    def alpha(x): return 1 / n_of_actions[x]

    # Play all arm once
    for i in range(k):
        reward = pull_arm(i, k_distribution)
        rewards.append(reward)

        n_of_actions[i] += 1
        values_of_actions[i] += (reward - values_of_actions[i]) * alpha(i)

        max_rewards.append(max(map(lambda x: x[0], k_distribution)))
        k_distribution = update_distribution(k_distribution, 0.001, 0.3)

    for t in range(1, steps):
        choice = choose_arm_ucb(values_of_actions, c, t, n_of_actions)
        reward = pull_arm(choice, k_distribution)
        rewards.append(reward)

        n_of_actions[choice] += 1
        values_of_actions[choice] += (reward -
                                      values_of_actions[choice]) * alpha(choice)

        max_rewards.append(max(map(lambda x: x[0], k_distribution)))
        k_distribution = update_distribution(k_distribution, 0.001, 0.3)

    return rewards, values_of_actions, max_rewards


if __name__ == "__main__":
    K = 10
    k_distribution = create_reward_distribution(K)

    experiment_configs = [
        (2000, 0),
        (2000, 1),
        (2000, 2),
        (2000, 3),
        (2000, 4),
        (2000, 5),
    ]

    for config in experiment_configs:
        rs, vs, mxrs = run_experiement(k_distribution, *config)

        cum_sum = 100 * np.cumsum(rs) / np.cumsum(mxrs)
        plt.plot(cum_sum, label=f"c - {config[1]}")
    plt.xscale('log')
    plt.title('Reward')
    plt.legend()
    plt.xlabel('Choice')
    plt.ylabel('% of Optimal Action')
    plt.ylim([0, 100])
    plt.show()
