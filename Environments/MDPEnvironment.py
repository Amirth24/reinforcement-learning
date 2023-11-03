from abc import ABC, abstractclassmethod

class MDPEnvironment(ABC):
    def __init__(self,  name: str = None):
        self.name= name
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def __mdp_step(self, state, action):
        self.action_history.append(action)
        self.state_history.append(state) 
        self.__apply_action(action) # current state is changed
        reward = self.__get_reward()
        self.reward_history.append(reward)

        return reward
    
    @classmethod
    @abstractclassmethod
    def __get_reward(self):
        pass

    @classmethod
    @abstractclassmethod
    def __apply_action(self, action):
        pass

    @classmethod
    @abstractclassmethod
    def step(self, action):
        pass
    