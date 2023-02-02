import gym
from gym import spaces
import numpy as np

class DamAgent(gym.Env):
    
    base_action_list = ['do_nothing','sell','buy']

    def __init__(self, data: np.ndarray, base_vol:int=1e5, base_height:int=30, is_tabular:bool=False, seed:int=None) -> None:
        
        #Configuring agent for tabular(discrete) or continous learning
        self.is_tabular = is_tabular
        
        #Setting Dam Max Volume
        self.base_vol = base_vol
        self.base_height = base_height
        
        #Declaring action space with 3 actions
        self.action_space = spaces.Discrete(3)
        
        #Declaring observation space with the the data we have
    
        if(self.is_tabular):
            low_range = np.full(shape=(data.shape[1]-1),fill_value=np.min(data[:,:-1],axis=0))
            high_range = np.full(shape=(data.shape[1]-1),fill_value=np.max(data[:,:-1],axis=0))
            low_range = np.concatenate([low_range,[0]])
            high_range = np.concatenate([high_range,[10]])
            self.observation_space = spaces.Box(low=low_range, high=high_range, dtype=np.int32)

        else:
            low_range = np.full(shape=(data.shape[1]),fill_value=-np.inf)
            high_range = np.full(shape=(data.shape[1]),fill_value=np.inf)
            self.observation_space = spaces.Box(low=low_range, high= high_range, dtype=np.float32)
        
        #Filling state space with the data
        
        
        if(self.is_tabular):
            self.state_space =np.concatenate((data[:,:-1], np.full((data.shape[0], 1),fill_value=5), np.full((data.shape[0], 1),fill_value=self.base_vol/2)), axis=1)
        else:
            self.state_space =np.concatenate((data[:,:-1], np.full((data.shape[0], 1),fill_value=0), np.full((data.shape[0], 1),fill_value=self.base_vol/2)), axis=1)
        
        #filling prices for reward generation

        self.prices = data[:,-1]

        #initialising base state
        
        self.seed = seed
        super().reset(seed=self.seed)
        np.random.seed(self.seed)

        self.reset()
        
        print(f"Initial state = {self.__get_obs()}")
        print("Environment initialised.")
        return

    def __discretize_vol(self, vol:int) -> int:
        out_vol = np.floor(vol/1e4).astype(int)
        return out_vol

    def __normalize_vol(self,vol:int) -> float:
        out_vol = float((vol-5e4)/5e4)
        return out_vol
    
    def __get_obs(self) -> int:
        self.state = self.state_space[self.clock]
        return self.state[:-1]
    
    def __get_info(self) -> dict:
        return {'cur_state':self.state[:-1], 'cur_price':self.price,'vol_lvl':self.state[-1],'clock':self.clock}

    def reset(self, do_random:bool=False) -> tuple:
        
        if(do_random):
            self.clock = np.random.randint(low=0,high=self.state_space.shape[0])
        else:
            self.clock = 0
        
        self.state=self.state_space[self.clock]
        self.state[-1] = self.base_vol/2
        self.price = self.prices[self.clock]

        if(self.is_tabular):
            self.state[-2] = self.__discretize_vol(self.state[-1])
        else:
            self.state[-2] = self.__normalize_vol(self.state[-1])
        return (self.__get_obs(), self.__get_info())

    def __convert_action_to_text(self,action:int) -> str:
        return self.base_action_list[action]    
    
    def __generate_reward(self, bool_buy:bool) -> float:
        
        max_delta = 5*3600
        
        cur_water_lvl = self.__get_info()['vol_lvl']
        
        if(bool_buy):
            eff_factor = 0.8
            # max_delta *= eff_factor

            if(cur_water_lvl+max_delta > self.base_vol):
                delta = self.base_vol - cur_water_lvl
            else:
                delta = max_delta

        else:
            eff_factor = 0.9
            
            if(cur_water_lvl-max_delta < 0):
                delta = -cur_water_lvl
            else:
                delta = -max_delta

        if(bool_buy):
            pot_energy = 1000*(1.25*delta)*9.81*self.base_height
        else:
            pot_energy = 1000*eff_factor*delta*9.81*self.base_height
        
        pot_energy /= 3.6e9
        reward = -pot_energy * self.price
        
        self.clock += 1
        self.price = self.prices[self.clock]
        self.state = self.state_space[self.clock]
        self.state[-1] = cur_water_lvl + delta
        if(self.is_tabular):
            self.state[-2] = self.__discretize_vol(self.state[-1])
        else:
            self.state[-2] = self.__normalize_vol(self.state[-1])
        
        return reward
    
    def step(self,action:int) -> tuple:
        if (not self.action_space.contains(action)):
            raise AssertionError("Invalid action value for agent step.")
        
        if self.clock < self.state_space.shape[0]-2:
            action_string = self.__convert_action_to_text(action)

            if(action_string == 'sell'):   
                if(self.is_tabular):
                    reward = self.__generate_reward(bool_buy=False)
                else:
                    reward = self.__generate_reward(bool_buy=False)
            
            elif(action_string == 'buy'):  
                if(self.is_tabular):
                    reward = self.__generate_reward(bool_buy=True)
                else:
                    reward = self.__generate_reward(bool_buy=True)

            else:
                reward = 0
                self.clock += 1
                self.state = self.state_space[self.clock]
                self.price = self.prices[self.clock]
                self.state[-1] = self.state_space[self.clock-1][-1]
                if(self.is_tabular):
                    self.state[-2] = self.__discretize_vol(self.state[-1]) 
                else:
                    self.state[-2] = self.__normalize_vol(self.state[-1])

            terminated = False

        else:
            reward = 0
            terminated = True
        
        observation = self.__get_obs()
        info = self.__get_info()

        return (observation, reward, terminated, False, info)


        
        

    

