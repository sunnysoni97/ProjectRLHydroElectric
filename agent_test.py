from Agent import DamAgent


if __name__ == "__main__":
    obj = DamAgent()
    
    tot_rew = 0
    mkt_price = 10.0
    
    print(obj.reset())

    for i in range(10):
        
        action = obj.action_space.sample()
        obs,reward,_,_,info = obj.step(action,mkt_price)
        print(f'Step : {obj.current_step}, Action : {action}, Reward : {reward}')
        
        tot_rew += reward
    
    print(f'Total rewards after 10 steps : {tot_rew}')
    print(info)
