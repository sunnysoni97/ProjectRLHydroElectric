import numpy as np
import Agent_Final
from collections import deque

if __name__ == "__main__":
    with open('data/train_data/train_discrete.npy','rb') as f:
        ary = np.load(f) 
    agent = Agent_Final.DamAgent(ary,is_tabular=False)
    print("current state : ")
    print(agent.reset())
    
    print("Doing 1 simul")

    terminated=False

    reward = 0.0
    mkt_price = 10.0

    while(not terminated):
        _,cur_reward,terminated,_,_ = agent.step(agent.action_space.sample(),mkt_price)
        reward += cur_reward
    
    print("Total reward in 1 simul : ", reward)
    
    '''
    a = deque([-np.inf],maxlen=100)
    print(a[0])
    '''
