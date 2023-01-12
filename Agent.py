import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DamAgent(gym.Env):
    
    base_action_list = ['do_nothing','sell_full','buy_full']

    def __init__(self,n_actions:int=3,n_vol_levels:int=2,render_mode=None) -> None:
        
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Dict({'vol':spaces.Discrete(n_vol_levels)})
        self.current_step = 0
        self.current_state = {'vol':self.observation_space['vol'].n-1}

        print("Agent initialised")
        print(f"Action space = {self.action_space}")
        print(f'Observation space = {self.observation_space}')
        return

    def __get_obs(self) -> int:
        return self.current_state['vol']
    
    def __get_info(self) -> dict:
        return {'current_state':self.current_state, 'current_step':self.current_step}

    def reset(self) -> tuple:
        self.current_state['vol'] = self.observation_space['vol'].n-1
        self.current_step = 0
        return (self.__get_obs(), self.__get_info())

    def __convert_action_to_text(self,action:int) -> str:
        return self.base_action_list[action]    
    
    def __generate_reward(self, delta_vol:float, market_price:float) -> float:
        return -delta_vol*market_price
    
    def step(self,action:int, market_price:float) -> tuple:
        if (not self.action_space.contains(action)):
            raise KeyError("Invalid action value for agent step.")
        
        action_string = self.__convert_action_to_text(action)

        reward = 0

        if(action_string == 'sell_full'):
            if(self.__get_obs() > 0):    
                delta_vol = float(0-self.__get_obs())
                self.current_state['vol'] = 0
                reward = self.__generate_reward(delta_vol,market_price)

        elif(action_string == 'buy_full'):
            if(self.__get_obs() < 1):    
                delta_vol = float(1-self.__get_obs())
                self.current_state['vol'] = 1
                reward = self.__generate_reward(delta_vol, market_price)

        self.current_step+=1
        observation = self.__get_obs()
        info = self.__get_info()
        return (observation, reward, False, False, info)
        

    

