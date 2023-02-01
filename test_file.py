from DDQN_Agent import DDQNAgent

if __name__ == "__main__":
    agent = DDQNAgent(mode='train',seed=123)
    rewards = agent.train_agent()
    print(f"rewards : {rewards}")

