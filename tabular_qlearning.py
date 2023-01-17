from Agent import DamAgent
import numpy as np
import os

class QLearnerTabular():
    def __init__(self, train_data, val_data, discount_rate=0.95) -> None:
        self.train_env = DamAgent(train_data)
        self.val_env = DamAgent(val_data)
        self.discount_rate = discount_rate
        self.action_space = self.train_env.action_space.n
        print("Q Learning Agent initialised")
        
    def __state_to_index(self, state_ar:np.ndarray) -> int:
        out = 0
        for i in range(len(state_ar)-1,-1,-1):
            out += state_ar[i]*pow(10,len(state_ar)-1-i)
        return int(out)
    
    def init_q_table(self) -> None:
        max_state = np.max(self.train_env.state_space[:,:-1],axis=0).astype(int)
        max_state = self.__state_to_index(max_state)
        self.q_table = np.zeros((max_state,self.action_space))
        print("Q table initialised")

    def train(self, simulations, learning_rate, epsilon = 0.05, epsilon_decay = 1000, adaptive_epsilon = False, 
              adapting_learning_rate = False):
        
        '''
        Params:
        
        simulations = number of episodes of a game to run
        learning_rate = learning rate for the update equation
        epsilon = epsilon value for epsilon-greedy algorithm
        epsilon_decay = number of full episodes (games) over which the epsilon value will decay to its final value
        adaptive_epsilon = boolean that indicates if the epsilon rate will decay over time or not
        adapting_learning_rate = boolean that indicates if the learning rate should be adaptive or not
        
        '''
        
        #Initialize variables that keep track of the rewards
        
        self.rewards = []
        self.average_rewards = []
        
        #Call the Q table function to create an initialized Q table
        self.init_q_table()
        
        #Set epsilon rate, epsilon decay and learning rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        #Set start epsilon, so here we want a starting exploration rate of 1
        self.epsilon_start = 1
        self.epsilon_end = 0.05
        
        #If we choose adaptive learning rate, we start with a value of 1 and decay it over time!
        if adapting_learning_rate:
            self.learning_rate = 1
        
        for i in range(simulations):
            
            if i%10 == 0:
                print(f'Please wait, the algorithm is learning! The current simulation is {i}')
            #Initialize the state
            state = self.train_env.reset()[0]   # reset returns a dict, need to take the 0th entry.
            state = state[:-1]
        
            #Set a variable that flags if an episode has terminated
            done = False
            
            #Set the rewards to 0
            total_rewards = 0
            
            #If adaptive epsilon rate
            if adaptive_epsilon:
                self.epsilon = np.interp(i, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
                
                #Logging just to check it decays as we want it to do, we just print out the first three statements
                if i % 500 == 0 and i <= 1500:
                    print(f"The current epsilon rate is {self.epsilon}")
                
            #Loop until an episode has terminated
            while not done:
                
                #Pick an action based on epsilon greedy
                
                '''
                ToDo: Write the if statement that picks a random action
                Tip: Make use of np.random.uniform() and the self.epsilon to make a decision!
                Tip: You can also make use of the method sample() of the self.env.action_space 
                    to generate a random action!
                '''
                
                #Pick random action
                if np.random.uniform(0,1) > 1-self.epsilon:
                    #This picks a random action from 0,1,2
                    action = self.train_env.action_space.sample()
                    
                    
                #Pick a greedy action
                else:
                    action = np.argmax(self.q_table[self.__state_to_index(state),:])
                    
                #Now sample the next_state, reward, done and info from the environment
                
                next_state, reward, terminated, truncated, info = self.train_env.step(action,state[0]) # step returns 5 outputs
                next_state = next_state[:-1]
                done =  terminated or truncated
                

                #Target value 
                Q_target = (reward + self.discount_rate*np.max(self.q_table[self.__state_to_index(next_state)]))
                
                #Calculate the Temporal difference error (delta)
                delta = self.learning_rate * (Q_target - self.q_table[self.__state_to_index(state), action])
                
                #Update the Q-value
                self.q_table[self.__state_to_index(state), action] = self.q_table[self.__state_to_index(state), action] + delta
                
                #Update the reward and the hyperparameters
                total_rewards += reward
                state = next_state
                
            
            if adapting_learning_rate:
                self.learning_rate = self.learning_rate/np.sqrt(i+1)
            
            self.rewards.append(total_rewards)
            
            #Calculate the average score over 100 episodes
            if i % 10 == 0:
                self.average_rewards.append(np.mean(self.rewards))
                print(f'Avg of Last 10 rewards : {self.average_rewards[-1]}')               
                #Initialize a new reward list, as otherwise the average values would reflect all rewards!
                self.rewards = []
        
        print('The simulation is done!')
        print(f'Rewards : {self.average_rewards}')


if __name__ == "__main__":
    train_file_path = os.path.join(os.path.dirname(__file__),'data/train_data/train_discrete.npy')
    with open(train_file_path,'rb') as f:
        training_data = np.load(f)

    val_file_path = os.path.join(os.path.dirname(__file__),'data/val_data/val_discrete.npy')
    with open(val_file_path,'rb') as f:
        validation_data = np.load(f)

    print("Training and Validation Data Loaded")

    QAgent = QLearnerTabular(train_data=training_data, val_data=validation_data)

    lr = 0.15
    simulations = 1000

    QAgent.train(simulations=simulations,learning_rate=lr)

    