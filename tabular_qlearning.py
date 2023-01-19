from Agent import DamAgent
import numpy as np
import os

class QLearnerTabular():
    def __init__(self, train_data:np.ndarray, val_data:np.ndarray, model_path:os.PathLike, discount_rate:float=0.95) -> None:
        self.train_env = DamAgent(train_data)
        self.val_env = DamAgent(val_data)
        self.discount_rate = discount_rate
        self.action_space = self.train_env.action_space.n
        self.model_path = model_path
        print("Q Learning Agent initialised")
        
    def __state_to_index(self, state_ar:np.ndarray) -> int:
        out = 0
        for i in range(len(state_ar)-1,-1,-1):
            out += state_ar[i]*pow(10,len(state_ar)-1-i)
        return int(out)
    
    def __init_q_table(self) -> None:
        max_state_price = np.max(np.max(self.train_env.state_space[:,0],axis=0)).astype(int)
        max_state_hour = np.max(np.max(self.train_env.state_space[:,1],axis=0)).astype(int)
        max_state_month = np.max(np.max(self.train_env.state_space[:,2],axis=0)).astype(int)
        max_state_vol = 10+1
        self.q_table = np.zeros((max_state_price,max_state_hour,max_state_month,max_state_vol,self.action_space))
        print(self.q_table.shape)
        print("Q table initialised")

    def __save_model_to_file(self) -> None:
        if(not os.path.exists(os.path.dirname(self.model_path))):
            os.makedirs(os.path.dirname(self.model_path))
        
        with open(self.model_path,'wb') as f:
            np.save(f,self.q_table)
        print(f"Model saved to file : {self.model_path}")    
    
    def __check_non_zeros(self,ar:np.ndarray) -> None:
        non_zero = ar[ar!=0]
        print(f"Number of non_zeroes : {non_zero.size}")

    def train(self, simulations:int, learning_rate:float, epsilon:float = 0.05, epsilon_decay:int = 1000, early_stopping_value:int=500, adaptive_epsilon:bool = False, 
              adapting_learning_rate:bool = False, early_stopping:bool=False) -> None:
        
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
        best_reward_train = -np.inf


        #Call the Q table function to create an initialized Q table
        self.__init_q_table()
        
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
        
        best_reward_val = self.validate()

        early_stop_ctr = 0

        for i in range(simulations):
            
            early_stop_ctr += 1

            if i%50 == 0:
                print(f'Please wait, the algorithm is learning! The current simulation is {i}')

            #Initialize the state
            state = self.train_env.reset()[0]   # reset returns a dict, need to take the 0th entry.
            state = state.astype(int)

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
                    action = np.argmax(self.q_table[state[0]-1,state[1]-1,state[2]-1,state[3]-1,:])
                    
                #Now sample the next_state, reward, done and info from the environment
                
                next_state, reward, terminated, truncated, info = self.train_env.step(action,state[0]) # step returns 5 outputs
                next_state = next_state.astype(int)
                done =  terminated or truncated
                

                #Target value 
                Q_target = (reward + self.discount_rate*np.max(self.q_table[next_state[0]-1,next_state[1]-1,next_state[2]-1,next_state[3]-1]))
                
                #Calculate the Temporal difference error (delta)
                delta = self.learning_rate * (Q_target - self.q_table[state[0]-1, state[1]-1, state[2]-1, state[3]-1, action])
                
                #Update the Q-value
                self.q_table[state[0]-1, state[1]-1, state[2]-1, state[3]-1, action] += delta
                
                #Update the reward and the hyperparameters
                total_rewards += reward
                state = next_state
                
            
            if adapting_learning_rate and i%10 == 0:
                self.learning_rate = self.learning_rate/np.sqrt((i/10)+1)
            
            self.rewards.append(total_rewards)
            
            #Calculate the average score over 100 episodes
            if i % 50 == 0:
                self.average_rewards.append(np.mean(self.rewards))
                print(f'Avg of Last 50 rewards : {self.average_rewards[-1]}')               
                #Initialize a new reward list, as otherwise the average values would reflect all rewards!
                self.rewards = []

            #save the best model
            if total_rewards > best_reward_train+1: 
                cur_reward_val = self.validate()
                if(cur_reward_val > best_reward_val+1):
                    self.__save_model_to_file()
                    best_reward_train = total_rewards
                    best_reward_val = cur_reward_val
                    early_stop_ctr = 0

            #check for early stopping
            if (early_stop_ctr == early_stopping_value):
                print("Early stopping due to no change!")
                break

        print('The simulation is done!')
        print(f'Average Rewards : {self.average_rewards}')

    def validate(self,load_model:bool=False) -> tuple:
        
        if(load_model):
            with open(self.model_path,'rb') as f:
                self.q_table = np.load(f)
        
        total_rewards = []
        total_actions = []
        done = False

        state = self.val_env.reset()[0]
        state = state.astype(int)
        action_list = []
        while(not done):
            action = np.argmax(self.q_table[state[0]-1, state[1]-1, state[2]-1,state[3]-1, :])
            action_list.append(action)
            mkt_price = state[0]
            state, reward, done, _, _ = self.val_env.step(action,mkt_price)
            action_taken = reward/mkt_price
            state = state.astype(int)
            total_rewards.append(reward)
            total_actions.append(action_taken)
        
        with open(os.path.join(os.path.dirname(self.model_path),'experiment_results.txt'),'wt') as f:
            f.write(f"Total reward on validation set : {np.sum(total_rewards)}\n")
            f.write(str(total_rewards))
            f.write(f"\nTotal action on validation set : {np.sum(total_actions)}\n")
            f.write(str(total_actions))
        
        return np.sum(total_rewards)


if __name__ == "__main__":
    train_file_path = os.path.join(os.path.dirname(__file__),'data/train_data/train_discrete.npy')
    with open(train_file_path,'rb') as f:
        training_data = np.load(f)

    val_file_path = os.path.join(os.path.dirname(__file__),'data/val_data/val_discrete.npy')
    with open(val_file_path,'rb') as f:
        validation_data = np.load(f)

    print("Training and Validation Data Loaded")

    def_model_path = os.path.join(os.path.dirname(__file__),'model/tabular_q/model.npy')
    
    QAgent = QLearnerTabular(train_data=training_data, val_data=validation_data, model_path=def_model_path, discount_rate=0.95)

    lr = 0.10
    simulations = 10000

    QAgent.train(simulations=simulations,learning_rate=lr,early_stopping_value=500,early_stopping=True)
    print("Validation of Best Model running : ")
    rewards = QAgent.validate(load_model=True)
    print(f"Total simulation reward on validation set = {rewards}")

    