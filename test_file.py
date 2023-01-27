import numpy as np
import Agent_Final
from collections import deque
from DDQN_Agent import DDQNAgent
import torch

if __name__ == "__main__":
    with open('data/train_data/train_discrete.npy','rb') as f:
        ary = np.load(f) 
    agent = Agent_Final.DamAgent(ary,is_tabular=False,seed=123)
    print("current state : ")
    print(agent.reset())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ddqn_agent = DDQNAgent(agent,device)

    rewards = ddqn_agent.train_agent()
