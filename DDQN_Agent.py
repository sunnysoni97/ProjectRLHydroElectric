import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

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
            new_obs, rew, terminated, truncated, _ = env.step(action,obs[0])
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
    
    def __init__(self, env, device, val_env:np.array=None, 
                 epsilon_decay:int=int(2e4), epsilon_start:float=1.0, epsilon_end:float=0.05, discount_rate:float=0.99, lr:float=5e-4, 
                 buffer_size:int=int(1e5), min_replay_size:int=int(1e4), replay_batch_size:int=168, update_freq_ratio:float=0.01, 
                 n_simuls:int=100, seed:int = None) -> None:
      
        self.env = env
        self.val_env = val_env
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        self.min_replay_size = min_replay_size
        self.replay_batch_size = replay_batch_size
        self.update_freq_ratio = update_freq_ratio
        self.n_simuls = n_simuls
        self.seed = seed

        self.replay_memory = ExperienceReplay(self.env, self.device, self.buffer_size, self.min_replay_size, seed = self.seed)
        self.online_network = DQN(self.env, self.learning_rate).to(self.device)
        
        
        #Solution:
        self.target_network = DQN(self.env, self.learning_rate).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

        #seeding random number generator for sampling
        self.seed = seed
        random.seed(self.seed)
        
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
    

    def train_agent(self) -> list:
        '''
        Returns:
        average_reward_list = a list of averaged rewards over 100 episodes of playing the game
        '''
        obs, _ = self.env.reset()
        average_reward_list = [-34000]
        episode_reward = 0.0
        
        steps_per_simul = self.env.state_space.shape[0]-1
        max_steps = self.n_simuls * steps_per_simul
        update_freq = int(steps_per_simul*self.update_freq_ratio)

        print(f"Update frequency : {update_freq}")

        for step in range(max_steps):
            
            action, epsilon = self.choose_action(step, obs)
        
            new_obs, rew, terminated, truncated, _ = self.env.step(action,obs[0])
            done = terminated or truncated        
            transition = (obs, action, rew, done, new_obs)
            self.replay_memory.add_data(transition)
            obs = new_obs
        
            episode_reward += rew
        
            if done:
            
                obs, _ = self.env.reset()
                self.replay_memory.add_reward(episode_reward)
                print(f'Buffer state : {self.replay_memory.reward_buffer}')
                #Reinitilize the reward to 0.0 after the game is over
                episode_reward = 0.0

            #Learn

            self.learn()

            #Calculate after each 100 episodes an average that will be added to the list
                    
            if (step) % update_freq == 0:
                
                average_reward_list.append(np.mean(self.replay_memory.reward_buffer))

            #update target network    
            if step % update_freq == 0:
                self.update_target_network()
        
            #Print some output
            if (step+1) % 1000 == 0:
                print(20*'--')
                print('Step', step)
                print('Epsilon', epsilon)
                print('Avg Rew', np.mean(self.replay_memory.reward_buffer))
                print()

        return average_reward_list        


    def validate(self) -> None:
    
        '''
        Params:
        step = the number of the step within the epsilon decay that is used for the epsilon value of epsilon-greedy
        seed = seed for random number generator for reproducibility
        '''
        
        print("Fudging implement this!")
        return None
        #Get the optimized strategy:
        done = False
        #Start the game
        state, _ = self.env.reset()
        while not done:
            #Pick the best action 
            action = self.choose_action(None, state, True)[0]
            next_state, rew, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated 
            state = next_state




