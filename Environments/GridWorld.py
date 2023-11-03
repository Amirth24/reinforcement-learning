"""
Grid world is an example of MDP in which each cell corresponds to the state of the environment
At each cell four actions are possible: north, south, east and west which deterministically causes 
agent to move from one cell to another in the respective direction on the grid
"""

import random
import sys
from typing import *
import pygame

from MDPEnvironment import MDPEnvironment

class GridWorld(MDPEnvironment):
    """
    This class represents the grid world. The grid world of any size can be created with an initial state.
    """
    __BLACK = (0, 0, 0)
    __RED = (255, 0, 0)
    

    def __init__(self, size: int,  current_pos: Tuple[int, int], name: str = None, speed=1):
        """Initializes the gridworld

        Args:
            size (int): The size of the grid world.
            current_pos (Tuple[int, int]): The starting position in which the agent starts
            name (str, optional): Name of the world. Defaults to None.
            speed (int, optional): Speed of the simulation. Defaults to 1.
        """
        self.grid_size = size
        self.current_pos = pygame.Vector2(current_pos)
        self.__fps = speed
        self.__t = 0
        self.__goal_position = None

        super().__init__(name)

        self.__init_graphics()

    def __init_graphics(self):
        """Initializes the window and graphics context
        """

        pygame.init()

        self.__window_height = 500
        self.__window_width = 500
        self.__agent_size = self.__window_height / self.grid_size



        self.__display = pygame.display.set_mode((self.__window_width, self.__window_height))

        self.__clock = pygame.time.Clock()


    
    def __display_agent(self):
        """Draws an agent on the screen. An agent is represented as a red box.
        """
        pygame.draw.rect(
            self.__display, 
            self.__RED,
            [
                self.current_pos.x * self.__agent_size, 
                self.current_pos.y * self.__agent_size, 
                (self.current_pos.x + 1) * self.__agent_size,
                (self.current_pos.y + 1) * self.__agent_size,
            ]
        )

    def __display_text(self):
        """Displays the important information about the environment
        """
        font = pygame.font.Font(
            pygame.font.get_default_font(),
            12
        )    

        avg_reward = sum(self.reward_history) / self.__t 

        text_ = font.render(f"Average reward: {avg_reward:.4f}", True, (255, 255, 255))
        text_rect = text_.get_rect()

        text_rect.center = (self.__window_width - 5 - text_rect.width / 2, 10)

        self.__display.blit(text_, text_rect)




    def draw(self):
        """Draws the grid world in the graphics context with the agent"""
        self.__display.fill(self.__BLACK)

        self.__clock.tick(self.__fps)

        self.__display_agent()
        self.__display_text()
        pygame.display.update()


    def step(self, action):
        """Takes an action on the current state

        Args:
            action (str): One of the possible actions 

        Returns:
            float: the reward for taking the action
        """
        self.__t += 1
        reward = super()._MDPEnvironment__mdp_step(self.current_pos, action)

        return reward, self.current_pos
        
    
    def _MDPEnvironment__get_reward(self):
        """The reward function that returns reward based on the current state.

        Returns:
            float: the reward value
        """
        if self.__goal_position and self.current_pos == self.__goal_position:
            return 10.0

        return -1.0
    

    def _MDPEnvironment__apply_action(self, action):
        """Applies action in the current to get to next state

        Args:
            action (str): one of the available actions

        Raises:
            Exception: If the action is not a valid one,  exception is raised.
        """
        match action:
            case 'left':
                if self.current_pos.x > 0: 
                    self.current_pos = self.current_pos + pygame.Vector2(-1, 0)  
            case 'right':
                if self.current_pos.x < self.grid_size : 
                    self.current_pos = self.current_pos + pygame.Vector2(1, 0) 
            case 'down':
                if self.current_pos.y < self.grid_size : 
                    self.current_pos = self.current_pos + pygame.Vector2(0, 1) 
            case 'up':
                if self.current_pos.y > 0: 
                    self.current_pos = self.current_pos + pygame.Vector2(0, -1)
            case _:
                raise Exception("Invalid Action")
        
    def set_goal_position(self, goal_position):
        """Set goal position that agent should reach"""
        self.__goal_position = goal_position



if __name__ == "__main__":
    gw = GridWorld(10,(5, 1), speed=30)
    print(pygame.font.get_fonts())
    gw.set_goal_position((6, 2))
    while True:

        reward, current_position = gw.step(random.choice(['left', 'right', 'up', 'down']))

        
        #event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        gw.draw()
