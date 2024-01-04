# Project RL - HydroElectric Dam

## Problem Description

**Simulation** : We are acting as a Hydro-Electric Dam operator. Our job is to make profit while trading electricity throughout the year. When we buy electricity, we use it to fill the dam. When we sell electricity, we empty the dam through the generator gates. Some real-world losses concerned with this process have to be integrated as well.

**RL Agent Training** : Accomplished using Classical Tabular based Q-learning and Double Deep-Q Networks.

## Steps for setting up the environment

For anaconda, use the command : conda env create -f environment.yml
in the main directory.

For vanilla python virtual environment or installation, you can also use : pip install -r requirements.txt

## File Descriptions

- Agent_Final.py : contains our self designed gym environment
- preprocess.py : contains code for preprocessing the xlsx files for our environment
- DDQN_Agent.py : contains the DDQN based RL agent
- ddqn_train.py : contains code for training and validating our DDQN agent on our custom environment
- tabular_qlearning.py : contains the Tabular Qlearning based RL agent as well as code for training and testing the models
- random_baseline.py : code for testing our agent with random actions on validation set

- exploratory_data_analysis.ipynb : EDA python notebooks
- Plots.ipynb : Python notebooks with the code used for plotting graphs for our report
- ddqn_plot_v_values.py : code for plotting the v-value plots

## Experiments

In all the experiment files, you can change the mode variable to 'train' to train your own models using the custom agent.

### For DDQN

1. Use ddqn_train.py for training/validation.
2. Look for parameters to configure using the class definition in DDQN_Agent.py.

### For Tabular

1. Use tabular_qlearning.py for training and validation.

### Random baseline

1. Use random_baseline.py for console output.

## Validation on standard environment

main.py makes use of our pretrained ddqn agent using the basic set to work on the validation set provided in validate.xlsx. The model used for this is provided in the same directory and should work out of the box.

## Results

The resulting plots have been provided in the `plots` directory (in the form of static images) as well as the `archived_results` directory (in the form of csv and text).
<br/>
<br/>

### Model and Data directory

If model and data directory do not exist, the validation will fail to work. Train your models first in these cases before trying to validate the custom environment.

#### Group (15) Members

Michal Butkiewicz, Sunny Soni, Andrzej Szczepura
