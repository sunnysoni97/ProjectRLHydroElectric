import numpy as np
import Agent_Final
from collections import deque
from DDQN_Agent import DDQNAgent
import torch
import preprocess

if __name__ == "__main__":
    # with open('data/train_data/train_big.npy','rb') as f:
    #     train_ary = np.load(f) 
    # with open('data/val_data/val_big.npy','rb') as f:
    #     val_ary = np.load(f) 
    # train_env = Agent_Final.DamAgent(train_ary,is_tabular=False,seed=123)
    # val_env = Agent_Final.DamAgent(val_ary)
    
    # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # print(train_env.reset())

    # ddqn_agent = DDQNAgent(train_env=train_env,device=device,val_env=val_env,buffer_size=20000,min_replay_size=10000,replay_batch_size=100,epsilon_decay=26000,update_freq_ratio=0.015)

    # rewards = ddqn_agent.train_agent()

    PP = preprocess.Preprocess_Continous()
    out = PP.preprocess_big('train.xlsx')
    empty_out = PP.preprocess_big('validate.xlsx',is_validate=True,train_values=out)
    out_dict,cols = PP.preprocess_small('train.xlsx')
    empty_dict,cols = PP.preprocess_small('validate.xlsx',is_validate=True,train_values=out_dict)
    PPD = preprocess.Preprocess_Tabular()
    PPD.preprocess_discrete('train.xlsx')
    PPD.preprocess_discrete('validate.xlsx',is_validate=True)
