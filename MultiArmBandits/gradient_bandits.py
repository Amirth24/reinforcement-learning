"""
Gradient bandits

In gradient bandits instead of calculating the expected reward for calculating the 
value of an action, a numerical preferences for each action , denoted by Ht(a) is learnt 
which has no interpretation in terms of reward. Only relative preferences of one action over
another is important. No matter how high a preference there is to an action it doesn't affect 
the probabilty of the action as the probablitity is determined by the softmax distribution.
"""
from typing import *

import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

from utils import create_reward_distribution, pull_arm, update_distribution


def update_preferences(a: int, old_preferences: np.array, reward: float, mean_reward: float, alpha: float) -> np.array:
    """Updates the preferences given the reward, mean_reward, and the step size alpha

    Args:
        a (int): The action that was chosen
        old_preferences (np.array): the preference that was used to choose the action a
        reward (float): reward obtained after choosing the action
        mean_reward (float): the expected reward of the action a
        alpha (float): the step size

    Returns:
        np.array: the updated preference 
    """
    proba = softmax(old_preferences)
    new_preferences = old_preferences
    new_preferences[a] += alpha * (reward - mean_reward) * (1 - proba[a])

    for i in range(len(old_preferences)):
        if i == a: pass
        new_preferences[i] -= alpha * (reward - mean_reward) * proba[i]
    
    # _1 = np.arange(0, len(old_preferences))
    # print(np.where(_1 == a, 1, 0), a)

    # return old_preferences + alpha * (reward - mean_reward) * (np.where(_1 == a, 1, 0) - proba)

    return new_preferences
    

def choose_action(preferences: np.array) -> int:
    return np.argmax(softmax(preferences))

def run_experiments(k_distribution: List[Tuple[float]], steps: int, alpha: float) -> Tuple[List[float], List[float]]:
    """Runs one full experitments with given number of steps
    Args:
        k_distribution: probability distribution for each action
        steps: number times choices to be made.
        alpha: step size parameter for the updation of preferences 
    Returns:
        rewards: list of reward obtained from each step
        max_rewards: list of maximum of means in the distribution
    """
    k = len(k_distribution)
    preferences = np.ones(k) / k
    values_of_actions = [0] * k
    rewards = []
    n_of_actions = [0]*k
    max_rewards = []
    # Play all arm once
    for i in range(k):
        reward = pull_arm(i, k_distribution)
        rewards.append(reward)

        n_of_actions[i] += 1
        values_of_actions[i] += (reward - values_of_actions[i]) * 0.9

        preferences = update_preferences(i, preferences, reward, values_of_actions[i], alpha )

        max_rewards.append(max(map(lambda x: x[0], k_distribution)))
        k_distribution = update_distribution(k_distribution, 0.001, 0.3)

    for t in range(1, steps):
        choice = choose_action(preferences)
        reward = pull_arm(choice, k_distribution)
        rewards.append(reward)
        n_of_actions[choice] += 1
        values_of_actions[choice] += (reward - values_of_actions[choice]) * 0.9

        preferences = update_preferences(i, preferences, reward, values_of_actions[choice], alpha )
        max_rewards.append(max(map(lambda x: x[0], k_distribution)))
        k_distribution = update_distribution(k_distribution, 0.001, 0.3)
        
    return rewards, max_rewards

if __name__ == "__main__":
    K = 10
    k_distribution = create_reward_distribution(K)
    print(k_distribution)
    experiment_configs = [
        (5000,0.01),
        (5000, 0.1),
        (5000, 0.01),
        (5000, 0.1),
    ]

    for config in experiment_configs:
        rs, mxrs = run_experiments(k_distribution, *config)

        cum_sum = 100 * np.cumsum(rs) / np.cumsum(mxrs)
        plt.plot(cum_sum, label=f"alpha - {config[1]}")
    plt.xscale("log")
    plt.title("Reward")
    plt.legend()
    plt.xlabel("Choice")
    plt.ylabel("% of Optimal Action")
    plt.ylim([0, 100])
    plt.show()
