import numpy as np
from math import floor
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

from Agent import DamAgent

#TODO: We need to incorporate the dataset at some point. For now, not sure where it should be loaded.

class DQN(nn.Module):
    
    def __init__(self, env, learning_rate):
        
        '''
        Params:
        env = environment that the agent needs to play
        learning_rate = learning rate used in the update
        
        '''
        
        super(DQN,self).__init__()
        input_features = env.observation_space.shape[0] #TODO: DamAgent returns the observation space as Dict so it will have to be flattened. However, we might consider just a Box
        action_space = env.action_space.n
        
        '''
        Make sure that the input features and the output features are in line with the environment that 
        the class takes as an input feature
        '''
        
        self.dense1 = nn.Linear(in_features = input_features, out_features = 128)
        self.dense2 = nn.Linear(in_features = 128, out_features = 64)
        self.dense3 = nn.Linear(in_features = 64, out_features = 32)
        self.dense4 = nn.Linear(in_features = 32, out_features = action_space)
        
        #Here we use ADAM, but you could also think of other algorithms such as RMSprob
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        
    def forward(self, x):
        
        '''
        Params:
        x = observation
        
        Important: We want to output a linear activation function as we need the q-values associated with each action
    
        '''
        
        #Solution:
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        x = self.dense4(x)
        
        return x
    
class ExperienceReplay:
    
    def __init__(self, env, buffer_size, min_replay_size = 1000, seed = 123):
        
        '''
        Params:
        env = environment that the agent needs to play
        buffer_size = max number of transitions that the experience replay buffer can store
        min_replay_size = min number of (random) transitions that the replay buffer needs to have when initialized
        seed = seed for random number generator for reproducibility
        '''
        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([0.0], maxlen = 100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print('Please wait, the experience replay buffer will be filled with random transitions')
                
        obs, _ = self.env.reset() #TODO: This applies to every reset. We need to set it to the first timestamp (?)
        for _ in range(self.min_replay_size):
            
        #Solution:
            action = env.action_space.sample()
            new_obs, rew, terminated, truncated, _ = env.step(action, obs[0])
            done = terminated or truncated

            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs
    
            if done:
                obs, _ = env.reset(seed=seed)
        
        print('Initialization with random transitions is done!')
      
          
    def add_data(self, data): 
        '''
        Params:
        data = relevant data of a transition, i.e. action, new_obs, reward, done
        '''
        self.replay_buffer.append(data)
            
    def sample(self, batch_size):
        
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
    
    def add_reward(self, reward):
        
        '''
        Params:
        reward = reward that the agent earned during an episode of a game
        '''
        
        self.reward_buffer.append(reward)
        

class DDQNAgent:
    
    def __init__(self, env, device, epsilon_decay, 
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, seed = 123):
        '''
        Params:
        env = environment
        device = set up to run CUDA operations
        epsilon_decay = Decay period until epsilon start -> epsilon end
        epsilon_start = starting value for the epsilon value
        epsilon_end = ending value for the epsilon value
        discount_rate = discount rate for future rewards
        lr = learning rate
        buffer_size = max number of transitions that the experience replay buffer can store
        seed = seed for random number generator for reproducibility
        '''
        self.env = env
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size
        
        self.replay_memory = ExperienceReplay(self.env, self.buffer_size, seed = seed)
        self.online_network = DQN(self.env, self.learning_rate).to(self.device)

        self.target_network = DQN(self.env, self.learning_rate).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        
    def choose_action(self, step, observation, greedy = False):
        
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
    
    
    def return_q_value(self, observation):
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
        
    def learn(self, batch_size):
        
        '''
        Params:
        batch_size = number of transitions that will be sampled
        '''
        
        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_memory.sample(batch_size)

        #Compute targets

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
        
        #Gradient descent to update the weights of the neural network
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()
        
    def update_target_network(self):
        
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    #TODO: We need to implement a similar function to the one below for validation
    '''
    The following method will let the DQNAgent play the game after it has worked 
    through the number of episodes for training
    '''
    # def play_game(self, step=1, seed=123):
        
    #     '''
    #     Params:
    #     step = the number of the step within the epsilon decay that is used for the epsilon value of epsilon-greedy
    #     seed = seed for random number generator for reproducibility
    #     '''
    #     #Get the optimized strategy:
    #     done = False
    #     #Reinitialize the game 
    #     self.env = gym.make(self.env_name, render_mode='human')
    #     #Start the game
    #     state, _ = self.env.reset()
    #     while not done:
    #         #Pick the best action 
    #         action = self.choose_action(step, state, True)[0]
    #         next_state, rew, terminated, truncated, _ = self.env.step(action)
    #         done = terminated or truncated 
    #         state = next_state
    #         #Pause to make it easier to watch
    #         time.sleep(0.05)
    #     #Close the pop-up window
    #     self.env.close()

def training_loop(env, agent, max_episodes, batch_size, seed=42):
    
    '''
    Params:
    env = name of the environment that the agent needs to play
    agent = which agent is used to train
    max_episodes = maximum number of games played
    batch_size =  number of transitions that will be sampled
    target = boolean variable indicating if a target network is used (this will be clear later)
    seed = seed for random number generator for reproducibility
    
    Returns:
    average_reward_list = a list of averaged rewards over 100 episodes of playing the game
    '''

    env.action_space.seed(seed)
    obs, _ = env.reset()
    average_reward_list = [0.0]
    episode_reward = 0.0
    
    for step in range(env.state_space.shape[0]*max_episodes):
        # print("Episode ", step)
        # print("obs: ", obs)
        # print("--"*20)
        
        action, epsilon = agent.choose_action(step, obs)
       
        new_obs, rew, terminated, truncated, _ = env.step(action, env.state_space[env.clock][0])
        # print("market price", env.state_space[step][0])
        # print("rew: ", rew)
        done = terminated or truncated        
        transition = (obs, action, rew, done, new_obs)
        agent.replay_memory.add_data(transition)
        obs = new_obs
    
        episode_reward += rew
    
        if done:
        
            obs, _ = env.reset()
            agent.replay_memory.add_reward(episode_reward)
            #Reinitilize the reward to 0.0 after the game is over
            episode_reward = 0.0

        #Learn

        agent.learn(batch_size)

        #Calculate after each 100 episodes an average that will be added to the list
                
        if (step+1) % 100 == 0:
            average_reward_list.append(np.mean(agent.replay_memory.reward_buffer))
            
        #Set the target_update_frequency
        target_update_frequency = 250
        if step % target_update_frequency == 0:
            agent.update_target_network()
    
        #Print some output
        if (step+1) % 10000 == 0:
            print(20*'--')
            print('Episode', floor(step/env.state_space.shape[0])+1)
            print('Epsilon', epsilon)
            print('Reward so far', episode_reward)
            print()

    return average_reward_list

if __name__ == "__main__":

    # Set the seed for reproducibility
    seed = 7
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    #Set the hyperparameters

    #Discount rate
    discount_rate = 0.99
    #That is the sample that we consider to update our algorithm
    batch_size = 32
    #Maximum number of transitions that we store in the buffer
    buffer_size = 50000
    #Minimum number of random transitions stored in the replay buffer
    min_replay_size = 1000
    #Starting value of epsilon
    epsilon_start = 1.0
    #End value (lowest value) of epsilon
    epsilon_end = 0.05
    #Decay period until epsilon start -> epsilon end
    epsilon_decay = 10000

    

    #Learning_rate
    lr = 5e-4

    with open('./data/train_data/train.npy', 'rb') as f:
        training_data = np.load(f)

    max_episodes = 5 #going 5 times through the entire dataset
    # max_episodes = 2500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dam = DamAgent(data = training_data)

    dagent = DDQNAgent(dam, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size)

    average_rewards_ddqn = training_loop(dam, dagent, max_episodes, batch_size) 
    