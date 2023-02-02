from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt
from DDQN_Agent import DDQNAgent
import os
import shutil

def train(n_simuls:int=5) -> None:
    print(f"Training for {n_simuls} simulations")
    train_agent = DDQNAgent(mode='train', seed=123, n_simuls=n_simuls)
    _, _ = train_agent.train_agent()
    print("-----------------TRAINING DONE----------------------")


parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='validate.xlsx') # Path to the excel file with the test data
args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data=args.excel_file)
total_reward = []
cumulative_reward = []



model_path = os.path.join(os.path.dirname(__file__),'best_online_net.bin')
pp_path = os.path.join(os.path.dirname(__file__),'train_mean_std.bin')

if(not os.path.exists(model_path) or not os.path.exists(pp_path)):
    train(1)
    new_model_path = os.path.join(os.path.dirname(__file__),'model/ddqn/best_online_net.bin')
    new_pp_path = os.path.join(os.path.dirname(__file__),'model/ddqn/train_mean_std.bin')
    shutil.copy2(src=new_model_path,dst=model_path)
    shutil.copy2(src=new_pp_path,dst=pp_path)

RL_agent = DDQNAgent(mode='validate_standard')

observation = env.observation()
for i in range(730*24 -1): # Loop through 2 years -> 730 days * 24 hours
    # Choose a random action between -1 (full capacity sell) and 1 (full capacity pump)
    # action = env.continuous_action_space.sample()
    # Or choose an action based on the observation using your RL agent!:
    action = RL_agent.act(observation)
    # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    next_observation, reward, terminated, truncated, info = env.step(action)
    total_reward.append(reward)
    cumulative_reward.append(sum(total_reward))

    done = terminated or truncated
    observation = next_observation

    if done:
        print('Total reward: ', sum(total_reward))
        # Plot the cumulative reward over time
        plt.plot(cumulative_reward)
        plt.xlabel('Time (Hours)')
        plt.show()




