from DDQN_Agent import DDQNAgent
import numpy as np

if __name__ == "__main__":
    
    n_simuls = 5
    print(f"Training for {n_simuls} simulations")
    agent = DDQNAgent(mode='train',dataset_big=False,seed=123,n_simuls=n_simuls)
    rewards_train, rewards_val = agent.train_agent()
    print(f'Max training reward : {np.max(rewards_train)}')
    print(f'Max validation reward : {np.max(rewards_val)}')

    print("-----------------VALIDATING-----------------")
    val_agent = DDQNAgent(mode='validate_custom',dataset_big=False,seed=123)
    total_rew = val_agent.validate_best()
    print(f'Total rew : {total_rew}')
    print("DONE")

