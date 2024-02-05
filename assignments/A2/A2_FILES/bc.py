import gym
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import optimizer

def train(learner, observations, actions, num_epochs=100):
    """Train function for learning a new policy using BC.
    
    Parameters:
        learner (Learner)
            A Learner object (policy)
        observations (list of numpy.ndarray)
            A list of numpy arrays of shape (7166, 11, ) 
        actions (list of numpy.ndarray)
            A list of numpy arrays of shape (7166, 3, )
        num_epochs (int)
            Number of epochs to run the train function for
    
    Returns:
        learner (Learner)
            A Learner object (policy)
    """
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)
    dataset = TensorDataset(torch.tensor(observations, dtype = torch.float32), torch.tensor(actions, dtype = torch.float32)) # Create your dataset
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True) # Create your dataloader
    
    # TODO: Complete the training loop here ###
    
    return learner