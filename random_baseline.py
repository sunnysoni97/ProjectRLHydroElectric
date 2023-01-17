from Agent import DamAgent
import numpy as np


if __name__ == "__main__":

    with open('./data/train_data/train.npy', 'rb') as f:
        training_data = np.load(f)

    env = DamAgent(data=training_data)
    
    episode_reward = 0

    for i in range(env.state_space.shape[0]):
        
        action = env.action_space.sample()
        mkt_price = env.state[0] # 1st element in state (price)
        obs,reward,_,_,info = env.step(action,mkt_price)
        
        episode_reward += reward

        if i % 5000 ==0:
            print("--"*20)
            print(f"Step {info['clock']}/{env.state_space.shape[0]}: Price: {mkt_price}, Action : {env.base_action_list[action]}, Vol lvl (after action): {obs[-1]}, Reward for that fookin choice: {reward}")
            print(f"Total reward so far: {round(episode_reward, 2)}$")
        # if i == 20:
        #     break
    print()
    print(f'Total reward of the episode : {round(episode_reward, 2)}$ (Big Benjis)')