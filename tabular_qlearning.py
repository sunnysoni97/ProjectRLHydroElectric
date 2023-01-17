from Agent import DamAgent
import numpy as np
import os

if __name__ == "__main__":
    train_file_path = os.path.join(os.path.dirname(__file__),'data/train_data/train_discrete.npy')
    with open(train_file_path,'rb') as f:
        training_data = np.load(f)

    val_file_path = os.path.join(os.path.dirname(__file__),'data/val_data/val_discrete.npy')
    with open(val_file_path,'rb') as f:
        validation_data = np.load(f)

    print("First row training data : ")
    print(training_data[0])
    print("First row validation data : ")
    print(validation_data[0])