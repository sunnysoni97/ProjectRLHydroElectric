import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from Agent_Final import DamAgent
from preprocess import Preprocess_Continous, preprocess_standard_observation
import pickle
import os
import pandas as pd

class DQN(nn.Module):
    
    def __init__(self, env, learning_rate:float) -> None:
        
        '''
        Params:
        env = environment that the agent needs to play
        learning_rate = learning rate used in the update
        '''
        
        super(DQN,self).__init__()
        input_features = env.observation_space.shape[0]
        action_space = env.action_space.n
        
        #initialising the layers of nn 
        
        self.dense1 = nn.Linear(in_features = input_features, out_features = 128)
        self.dense2 = nn.Linear(in_features = 128, out_features = 64)
        self.dense3 = nn.Linear(in_features = 64, out_features = 32)
        self.dense4 = nn.Linear(in_features = 32, out_features = action_space)
        
        #using adam optimiser
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        '''
        Params:
        x = observation
        '''
        
        #forward pass for nn

        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        x = self.dense4(x)
        
        return x

class ExperienceReplay:
    
    def __init__(self, env, device, buffer_size:int, min_replay_size:int, seed:int = None) -> None:
        
        '''
        Params:
        env = environment that the agent needs to play
        device = torch device to use
        buffer_size = max number of transitions that the experience replay buffer can store
        min_replay_size = min number of (random) transitions that the replay buffer needs to have when initialized
        seed = seed for random number generator for reproducibility
        '''

        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([-34000.0], maxlen = 100)
        self.device = device
        
        print('Please wait, the experience replay buffer will be filled with random transitions')

        #initialising replay buffer with random transitions

        obs, _ = self.env.reset(do_random=True)
        
        for _ in range(self.min_replay_size):
            
            action = env.action_space.sample()
            new_obs, rew, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs
    
            if done:
                obs, _ = env.reset(do_random=True)
        
        print('Initialization with random transitions is done!')
      
        #seeding random number generator for sampling
        self.seed = seed
        random.seed(self.seed)
          
    def add_data(self, data:tuple) -> None: 
        
        '''
        Params:
        data = relevant data of a transition, i.e. action, new_obs, reward, done
        '''

        #adding new transition to replay buffer

        self.replay_buffer.append(data)
            
    def sample(self, batch_size:int) -> tuple:
        
        '''
        Params:
        batch_size = number of transitions that will be sampled
        
        Returns:
        tensor of observations, actions, rewards, done (boolean) and next observation 
        '''
        
        transitions = random.sample(self.replay_buffer, batch_size)

        #Solution
        observations = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_observations = np.asarray([t[4] for t in transitions])

        #PyTorch needs these arrays as tensors!, don't forget to specify the device! (cpu / GPU)
        observations_t = torch.as_tensor(observations, dtype = torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype = torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype = torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype = torch.float32, device=self.device).unsqueeze(-1)
        new_observations_t = torch.as_tensor(new_observations, dtype = torch.float32, device=self.device)
        
        return observations_t, actions_t, rewards_t, dones_t, new_observations_t
    
    def add_reward(self, reward:float) -> None:
        
        '''
        Params:
        reward = reward that the agent earned during an episode of a game
        '''

        #adding reward to reward buffer        

        self.reward_buffer.append(reward)
        

class DDQNAgent:
    
    def __init__(self, mode:str='validate_custom', dataset_big:bool=False, train_file_path:str='train.xlsx', val_file_path:str='validate.xlsx', device:str='cpu', 
                 epsilon_decay:int=int(2e4), epsilon_start:float=1.0, epsilon_end:float=0.05, discount_rate:float=0.99, lr:float=5e-4, 
                 buffer_size:int=int(2e4), min_replay_size:int=int(1e4), replay_batch_size:int=100, update_freq_ratio:float=0.015, val_check_step:int=1000,
                 n_simuls:int=5, seed:int = None) -> None:
      
        self.model_base_path = os.path.join(os.path.dirname(__file__),'model/ddqn/')
        if(not os.path.exists(self.model_base_path)):
            os.makedirs(self.model_base_path)
        
        data_base_path = os.path.join(os.path.dirname(__file__),'data/')

        self.device = device

        self.learning_rate = lr

        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

        if(mode=='train'):
            
            #Preprocessing data for training and validation
            PP = Preprocess_Continous()
            if(dataset_big):
                train_dict = PP.preprocess_big(train_file_path)
                PP.preprocess_big(val_file_path,is_validate=True,train_values=train_dict)
            
            else:
                train_dict,_ = PP.preprocess_small(train_file_path)
                PP.preprocess_small(val_file_path,is_validate=True,train_values=train_dict)
            
            with open(os.path.join(self.model_base_path,'train_mean_std.bin'),'wb') as f:
                    pickle.dump(train_dict,f)
            print("training dataset preprocessing values saved to disk")
            
            if(not os.path.exists(data_base_path)):
                raise IOError("Processed Data Files do not exist!!!")
            
            feature_set = 'big' if dataset_big else 'small'
            with open(os.path.join(data_base_path,f'train_data/train_{feature_set}.npy'),'rb') as f:
                train_ary = np.load(f)
            
            with open(os.path.join(data_base_path,f'val_data/val_{feature_set}.npy'),'rb') as f:
                val_ary = np.load(f)

            train_env = DamAgent(train_ary,seed=seed)
            val_env = DamAgent(val_ary,seed=seed)
            val_train_env = DamAgent(train_ary,seed=seed)


            self.env = train_env
            self.val_env = val_env
            self.val_train_env = val_train_env

            self.discount_rate = discount_rate
            
            self.buffer_size = buffer_size
            self.min_replay_size = min_replay_size
            self.replay_batch_size = replay_batch_size

            self.update_freq_ratio = update_freq_ratio
            self.val_check_step = val_check_step
            self.n_simuls = n_simuls
            
            self.seed = seed
            random.seed(self.seed)

            self.replay_memory = ExperienceReplay(self.env, self.device, self.buffer_size, self.min_replay_size, seed = self.seed)

            self.online_network = DQN(self.env, self.learning_rate).to(self.device)
        
            self.target_network = DQN(self.env, self.learning_rate).to(self.device)
            self.target_network.load_state_dict(self.online_network.state_dict())

        elif(mode=='validate_custom'):
            feature_set = 'big' if dataset_big else 'small'
            
            with open(os.path.join(data_base_path,f'val_data/val_{feature_set}.npy'),'rb') as f:
                val_ary = np.load(f)

            val_env = DamAgent(val_ary,seed=seed)

            self.val_env = val_env
            
            self.online_network = DQN(self.val_env, self.learning_rate).to(self.device)
            self.load_network_from_disk()
            
        elif(mode=='validate_standard'):
            
            self.model_base_path = os.path.join(os.path.dirname(__file__))
            
            preprocess_dict_path = os.path.join(os.path.dirname(__file__),'train_mean_std.bin')
            if(not os.path.exists(preprocess_dict_path)):
                raise IOError("Preprocessing value dictionary not present in the folder!!!")

            with open(preprocess_dict_path,'rb') as f:
                self.pp_values_dict = pickle.load(f)
            
            random_data = np.zeros((10,4))

            val_env = DamAgent(random_data)

            self.online_network = DQN(val_env, self.learning_rate).to(self.device)
            self.load_network_from_disk()

        else:
            raise ValueError("Invalid Mode for initialising DDQN agent!!!")
            
        
    def act(self,obs_standard) -> float:
        new_obs = preprocess_standard_observation(obs=obs_standard,pp_values_dict=self.pp_values_dict)
        action_index,_ = self.choose_action(None,new_obs,True)
        if(action_index == 1):
            out_action = -1.0
        elif(action_index == 2):
            out_action = 1.0
        else:
            out_action = 0.0
        
        return out_action
    
    
    def choose_action(self, step:int, observation, greedy = False) -> tuple:
        
        '''
        Params:
        step = the specific step number 
        observation = observation input
        greedy = boolean that
        
        Returns:
        action: action chosen (either random or greedy)
        epsilon: the epsilon value that was used 
        '''
        
        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        random_sample = random.random()
    
        if (random_sample <= epsilon) and not greedy:
            #Random action
            action = self.env.action_space.sample()
        
        else:
            #Greedy action
            obs_t = torch.as_tensor(observation, dtype = torch.float32, device=self.device)
            q_values = self.online_network(obs_t.unsqueeze(0))
        
            max_q_index = torch.argmax(q_values, dim = 1)[0]
            action = max_q_index.detach().item()
        
        return action, epsilon
    
    
    def return_q_value(self, observation) -> float:
        
        '''
        Params:
        observation = input value of the state the agent is in
        
        Returns:
        maximum q value 
        '''
        #We will need this function later for plotting the 3D graph
        
        obs_t = torch.as_tensor(observation, dtype = torch.float32, device=self.device)
        q_values = self.online_network(obs_t.unsqueeze(0))
        
        return torch.max(q_values).item()
        
    def learn(self) -> None:
        
        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_memory.sample(self.replay_batch_size)

        #Compute targets, note that we use the same neural network to do both! This will be changed later!

        target_q_values = self.target_network(new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        #scaling rewards to small values
        rewards_t = rewards_t/100
        
        targets = rewards_t + self.discount_rate * (1-dones_t) * max_target_q_values

        #Compute loss

        q_values = self.online_network(observations_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        #Loss, here we take the huber loss!

        loss = F.smooth_l1_loss(action_q_values, targets)
        
        #Uncomment the following code to use the MSE loss instead!
        #loss = F.mse_loss(action_q_values, targets)
        
        #Gradient descent to update the weights of the neural networ
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()
        
    def update_target_network(self) -> None:
        
        '''
        ToDO: 
        Complete the method which updates the target network with the parameters of the online network
        Hint: use the load_state_dict method!
        '''
    
        #Solution:
        
        self.target_network.load_state_dict(self.online_network.state_dict())
    

    def train_agent(self) -> tuple:
        '''
        Returns:
        average_reward_list = a list of averaged rewards over 100 episodes of playing the game
        '''
        obs, _ = self.env.reset(do_random=False)
        
        reward_list_train = []
        reward_list_val = []
        step_list = []
        episode_reward = 0.0

        best_val_score = -np.inf
        
        steps_per_simul = self.env.state_space.shape[0]-1
        max_steps = self.n_simuls * steps_per_simul
        update_freq = int(steps_per_simul*self.update_freq_ratio)

        print(f"Update frequency : {update_freq}")

        for step in range(max_steps):
            
            action, epsilon = self.choose_action(step, obs)
        
            new_obs, rew, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated        
            transition = (obs, action, rew, done, new_obs)
            self.replay_memory.add_data(transition)
            obs = new_obs
        
            episode_reward += rew
        
            if done:
            
                obs, _ = self.env.reset(do_random=False)
                self.replay_memory.add_reward(episode_reward)
                print(f'Buffer state : {self.replay_memory.reward_buffer}')
                print(f'Last Train Score : {reward_list_train[-1]}')
                print(f'Last Validate Score : {reward_list_val[-1]}')
                #Reinitilize the reward to 0.0 after the game is over
                episode_reward = 0.0

            #Learn

            self.learn()

            #Calculate after each 100 episodes an average that will be added to the list
                    
            if((step+1) % self.val_check_step == 0 and (step+1) % 2000 != 0):
                
                reward_val,_,_ = self.validate()

                if(reward_val>best_val_score):
                    print(f"Best validation score : {reward_val}")
                    self.save_network_to_disk()
                    best_val_score=reward_val

            
            if (step+1) % 2000 == 0:
                reward_train,_,_ = self.validate(is_train=True)
                reward_val,_,_ = self.validate()

                if(reward_val>best_val_score):
                    print(f"Best validation score : {reward_val}")
                    self.save_network_to_disk()
                    best_val_score=reward_val

                reward_list_train.append(reward_train)
                reward_list_val.append(reward_val)
                step_list.append(step+1)

            #update target network    
            if step % update_freq == 0:
                self.update_target_network()
        
            #Print some output
            if (step+1) % 1000 == 0:
                print(20*'--')
                print('Step', step)
                print('Avg Rew', np.mean(self.replay_memory.reward_buffer))


        df_dict = {'train_reward':reward_list_train,'val_reward':reward_list_val,'step':step_list}
        df = pd.DataFrame(data=df_dict)
        df.to_csv(os.path.join(self.model_base_path,'train_val_rewards.csv'),index=False)

        return reward_list_train,reward_list_val 


    def validate(self,is_train:bool=False) -> tuple:
            
        if(is_train):
            state,info = self.val_train_env.reset()
        else:
            state,info = self.val_env.reset()
        
        episode_rew = 0
        rewards_at_steps = []
        actions_at_steps = []
        cur_price = info['cur_price']

        done = False
        while(not done):
            action = self.choose_action(None, state, True)[0]
            
            if(is_train):
                next_state, rew, terminated, truncated, info = self.val_train_env.step(action)
            else:
                next_state, rew, terminated, truncated, info = self.val_env.step(action)
            
            episode_rew += rew
            rewards_at_steps.append(rew)
            
            cur_action = rew/cur_price
            actions_at_steps.append(cur_action)
            
            done = terminated or truncated 
            if(not done):
                cur_price = info['cur_price']
            state = next_state

        return episode_rew, rewards_at_steps, actions_at_steps

    def validate_best(self) -> float:
        total_rew, reward_list, action_list = self.validate()
        
        df_dict = {'reward':reward_list,'action':action_list,'step':[x for x in range(len(reward_list))]}
        df = pd.DataFrame(data=df_dict)
        df.to_csv(os.path.join(self.model_base_path,'cummulative_rewards.csv'),index=False)

        return total_rew
    
    def save_network_to_disk(self) -> None:
        with open(os.path.join(self.model_base_path,'best_online_net.bin'),'wb') as f:
            torch.save(self.online_network.state_dict(),f)
        print(f"Best network saved to {self.model_base_path} : best_online_net.bin")

    def load_network_from_disk(self) -> None:
        model_path = os.path.join(self.model_base_path,'best_online_net.bin')
        
        if(not os.path.exists(model_path)):
            raise IOError("Best model doesn't exist on disk!!!")
        
        with open(model_path,'rb') as f:
            disk_state_dict = torch.load(f)
            self.online_network.load_state_dict(disk_state_dict)
        
        print(f"Best network loaded from {self.model_base_path} : best_online_net.bin")
        



