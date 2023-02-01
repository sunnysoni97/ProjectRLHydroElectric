from Agent_Final import DamAgent
import numpy as np
from preprocess import Preprocess_Tabular
import os

if __name__ == "__main__":

    PP = Preprocess_Tabular()
    
    #ASSUMING validate.xlsx has the same structure(not data values) as provided to students,
    #otherwise might break

    PP.preprocess_discrete('validate.xlsx',is_validate=True)

    val_data_path = os.path.join(os.path.dirname(__file__),'data/val_data/val_discrete.npy')

    with open(val_data_path, 'rb') as f:
        val_data = np.load(f)

    env = DamAgent(data=val_data)
    
    episode_reward = 0

    for i in range(env.state_space.shape[0]):
        
        action = env.action_space.sample()
        obs,reward,_,_,info = env.step(action)
        
        episode_reward += reward

        if i % 5000 ==0:
            print(f"Step number : {i+1}")
            print(f"Total reward so far: {round(episode_reward, 2)}$")
    
    print(f'Total reward of the episode : {round(episode_reward, 2)}$ (Big Benjis)')