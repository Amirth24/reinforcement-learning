from typing import * 
import random



def create_reward_distribution(k: int) -> List[Tuple[float, float]]:
    """Creates a probability distribution for each action. 
    Returns a List of pairs of mu(mean) and sigma(var) with some randomness
    Args:
        k: number of arms/choices
    Returns:
        reward_distribution: list of pairs of mean and variance for normal distribution
    """
    return [(x + random.random(), random.random()) for x in range(k)]


def pull_arm(k: int, distribution: List[Tuple[float, float]]) -> float: 
    """Pulls an arm k and gives the reward
    Args:
        k : index of arm of choice
        distribution: list of tuples with normal distribution parameters(mu, sigma)

    Returns:
        reward: reward for the chosen arm
    """
    mu, sigma = distribution[k]

    return random.normalvariate(mu, sigma)


def choose_optimally(action_values: List[float], epsilon: float) -> int:
    """Choose a epsilon greedy action
    Args:
        action_values: Values of each action
        epsilon: probability of exploring non greedy actions
    Returns:
        action: action n chosen 
    """
    if random.random() < epsilon:
        return random.choice(range(len(action_values)))
    else:
        return action_values.index(max(action_values))

def update_distribution(distribution: List[Tuple[float, float]], strength: float, max_reward: float) ->  List[Tuple[float, float]]:
    """Updates the reward distribution with some randomness
    Args:
        distribution: list of tuples with normal distribution parameters(mu, sigma)
        strength: strength of change in reward
        max_reward: sets the limit of mean of x
    Returns:
        updated_distribution: distributions with updated mean and variance 
    """
    def update_val(x):
        if x[0] >  max_reward:
            return (strength * random.random()) + x[0] - max_reward * strength, x[1] * random.random()
        else:
            return (strength * random.random() ) + x[0], x[1] * random.random()
    return list(map(update_val, distribution))