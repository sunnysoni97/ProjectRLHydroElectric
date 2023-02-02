from DDQN_Agent import DDQNAgent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#function to convert plotting features to observation

# def make_observation(price:float,hour:int, month:int,vol_lvl:int) -> np.ndarray:
def make_observation(price:float,hour:int, month:int,vol:float) -> np.ndarray:
    low_consts = [-1.6,-1.66]
    high_consts = [1.6,1.66]
    month_values = np.linspace(start=low_consts[0],stop=high_consts[0],num=12)
    # vol_values = np.linspace(start=low_consts[1],stop=high_consts[1],num=10)
    hour_values = np.linspace(start=low_consts[1],stop=high_consts[1],num=24)
    month = month_values[month-1]
    # vol = vol_values[vol_lvl-1]
    hour = hour_values[hour-1]
    obs = np.array(object=[price,hour,month,vol])
    return obs

if __name__ == "__main__":
    
    #loading validation agent

    val_agent = DDQNAgent(mode='validate_custom')

    #setting range for price and hour

    low = [-2.0,-1.0]
    high = [2.0,1.0]

    #setting month and vol as consts
    
    month = 6
    hour = 19

    #making bins for smoothing
    
    bin_size = 20
    bin_price = np.linspace(low[0], high[0], bin_size)
    # bin_hour = np.linspace(low[1], high[1], bin_size)
    bin_vol = np.linspace(low[1], high[1], bin_size)

    X, Y = np.meshgrid(bin_price, bin_vol)
    Z = np.zeros((len(X), len(Y)))

    #filling V values for our features

    for i in range(len(X)):
        for j in range(len(Y)):
            obs = make_observation(price=X[0][i], vol=Y[j][0],month=month,hour=hour)
            Z[i][j] = val_agent.return_q_value(obs)
    fig = plt.figure(figsize =(10,10))
    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    ax.set_xlabel('Market Price (Z-Normalised)', fontsize = 14)
    ax.set_ylabel('Volume (Z-Normalised)', fontsize = 14)
    ax.set_zlabel('Value Function', fontsize = 14)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Value function vs Price and Hour', fontsize = 18)
    
    #saving to disk
    
    plt.savefig(f'v_vs_price+vol_at_{hour}_hour.png')

