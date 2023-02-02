from DDQN_Agent import DDQNAgent
import numpy as np

if __name__ == "__main__":
    
    #change this to 'train' to train new model
    mode='val'

    #change this to 'true' to test out our engineered dataset (LOWER performance)
    dataset_big = False

    if(mode=='train'):
        n_simuls = 5
        print(f"Training for {n_simuls} simulations")
        agent = DDQNAgent(mode='train', dataset_big=dataset_big, seed=123, n_simuls=n_simuls, replay_batch_size=100)
        rewards_train, rewards_val = agent.train_agent()
        print(f'Max training reward : {np.max(rewards_train)}')
        print(f'Max validation reward : {np.max(rewards_val)}')

    print("-----------------VALIDATING-----------------")
    val_agent = DDQNAgent(mode='validate_custom', dataset_big=dataset_big, seed=123)
    total_rew = val_agent.validate_best()
    print(f'Total rew : {total_rew}')
    print("DONE")

