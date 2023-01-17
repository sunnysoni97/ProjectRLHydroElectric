import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DamAgent(gym.Env):
    
    base_action_list = ['do_nothing','sell','buy']

    def __init__(self, data: np.ndarray, base_vol:int=1e5, base_height:int=30, n_actions:int=3) -> None:
        
        self.base_vol = base_vol/2
        self.base_height = base_height
        
        self.action_space = spaces.Discrete(n_actions)
        # Features: 'price', 'bollinger_up', 'bollinger_middle', 'bollinger_down', '120h / 5day EMA', '480h / 20day EMA', '2400h / 100day EMA', '120h / 5day ATR', 'vol_lvl'
        self.observation_space = spaces.Box(low=np.zeros(9), high= np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, self.base_vol]), dtype=np.float32)
        
        #adding a column of zeros for storing the water volume
        self.state_space =np.concatenate((data, np.zeros((data.shape[0], 1))), axis=1)
        self.state_space[0][-1] = self.base_vol
        self.clock = 0
        self.state = self.state_space[self.clock]

        print("Environment initialised.")
        return

    def __get_obs(self) -> int:
        self.state = self.state_space[self.clock]
        self.state[-1] = self.state_space[self.clock-1][-1]
        return self.state
    
    def __get_info(self) -> dict:
        return {'state':self.state, 'clock':self.clock}

    def reset(self) -> tuple:
        self.state=self.state_space[0]
        self.clock = 0
        return (self.__get_obs(), self.__get_info())

    def __convert_action_to_text(self,action:int) -> str:
        return self.base_action_list[action]    
    
    def __generate_reward(self, bool_buy:bool, market_price:float) -> float:
        
        max_delta = 5*3600
        
        if(bool_buy):
            eff_factor = 0.8
            max_delta *= eff_factor

            if(self.__get_obs()[-1]+max_delta > self.base_vol):
                delta = self.base_vol - self.__get_obs()[-1]
            else:
                delta = max_delta

        else:
            eff_factor = 0.9
            
            if(self.__get_obs()[-1]-max_delta < 0):
                delta = -self.__get_obs()[-1]
            else:
                delta = -max_delta

        if(bool_buy):
            pot_energy = 1000*(1.2*delta)*9.81*self.base_height
        else:
            pot_energy = 1000*eff_factor*delta*9.81*self.base_height
        
        pot_energy /= 3.6e9
        reward = -pot_energy * market_price
        self.state[-1] += delta
        
        return reward
    
    def step(self,action:int, market_price:float) -> tuple:
        if (not self.action_space.contains(action)):
            raise AssertionError("Invalid action value for agent step.")
        
        action_string = self.__convert_action_to_text(action)

        reward = 0

        if(action_string == 'sell'):   
            reward = self.__generate_reward(bool_buy=False,market_price=market_price)


        elif(action_string == 'buy'):  
            reward = self.__generate_reward(bool_buy=True, market_price=market_price)

        if self.clock < self.state_space.shape[0]-1:
            terminated = False
            self.clock += 1
            observation = self.__get_obs()
            info = self.__get_info()
        else:
            observation = self.__get_obs()
            info = self.__get_info()
            terminated = True

        return (observation, reward, terminated, False, info)
        

    

