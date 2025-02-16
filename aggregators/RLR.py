
from utils.typings import Errors, Accuracy, AttackSuccesRate
from experiment.AggregatorConfig import AggregatorConfig
from torch import nn
from client import Client
from logger import logPrint
from typing import List,Tuple
import torch
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
from datasetLoaders.DatasetLoader import DatasetLoader
from torch import Tensor
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from scipy.ndimage import rotate, zoom, shift
import cv2
import random

import torch.nn.functional as F
from copy import deepcopy
import gc




class RLRAggregator(Aggregator):
    """
    Federated Averaging (FedAvg) with the Robust Learning Rate (RLR) 
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
    
        useAsyncClients: bool = False,
    
    ):
        super().__init__(clients, model, config, useAsyncClients)
        
        # Initialize the AggregatorConfig object with the necessary attributes
        self.config = config
        self.config.noise = 0.0  # Set the noise attribute as needed
        self.config.clip = 1   # Set the clip attribute as needed
        self.theta = 0.0  # Threshold for robust learning rate
        self.learning_rate = 1 # Server learning rate

        
       

    def trainAndTest(self, testDataset: DatasetInterface) -> Tuple[Errors, AttackSuccesRate]:
        roundsError = torch.zeros(self.rounds)
        roundsattack_success_rate = torch.zeros(self.rounds)
        # Adding dynamic learning rate adjustment
        learning_rate_scheduler = lambda r: 1 / (1 + 0.001)  # Customize as needed
        for r in range(self.rounds):
            self.learning_rate = learning_rate_scheduler(r)
            gc.collect()
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()
            chosen_clients = [self.clients[i] for i in self.chosen_indices]
            self.model = self.aggregate_with_RLR(chosen_clients, models)  # Get the global model with RLR
            # Compute error and attack success rate
            error, success_rate = self.test(testDataset)
            roundsError[r] = torch.tensor(error, dtype=torch.float32)
            roundsattack_success_rate[r] = torch.tensor(success_rate, dtype=torch.float32)
            #attack_success_variance[r] = torch.tensor(variance, dtype=torch.float32)

            print(f'Accuracy of the model on clean images: {roundsError[r]:.2f}%')
            print(f'Attack success rate: {roundsattack_success_rate[r]:.2f}%')
            #print(f'attack_success_variance: {attack_success_variance[r]:.2f}%')

        return roundsError, roundsattack_success_rate
    
    

    
    def aggregate_with_RLR(self, chosen_clients, models) -> torch.nn.Module:
        """
        Aggregates client models using the Robust Learning Rate (RLR) method.

        Parameters:
        - chosen_clients: List of selected client objects for the aggregation.
        - models: A dictionary containing client models.

        Returns:
        - An aggregated global model.
        """
        agent_updates_dict = self._extract_client_updates(chosen_clients, models)

        # Compute robust learning rate
        lr_vector = self.compute_robustLR(agent_updates_dict)
        
        aggregated_updates = {name: torch.zeros_like(param) 
                              for name, param in self.model.named_parameters()}
        
        # Aggregate updates
        for update_dict in agent_updates_dict.values():
            for name, update_tensor in update_dict.items():
                aggregated_updates[name] += update_tensor
        
        # Average the updates
        for name in aggregated_updates:
            aggregated_updates[name] /= len(agent_updates_dict)
            
        # Clipping the aggregated updates to prevent excessively large updates
        for name in aggregated_updates:
            aggregated_updates[name] = torch.clamp(aggregated_updates[name], min=-self.config.clip, max=self.config.clip)
        
        # Apply updates to the global model with the adjusted learning rate
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param += self.learning_rate * lr_vector[name] * aggregated_updates[name]
                
        return self.model

        



    def compute_robustLR(self, agent_updates_dict):
        aggregated_signs = {name: torch.zeros_like(param) 
                            for name, param in self.model.named_parameters()}

        for update_dict in agent_updates_dict.values():
            for name, update_tensor in update_dict.items():
                aggregated_signs[name] += torch.sign(update_tensor)

        lr_vector = {}
        # Adjust the learning rate to be more adaptive
        for name, sign_tensor in aggregated_signs.items():
            non_zero_elements = torch.nonzero(torch.abs(sign_tensor), as_tuple=True)
            adaptive_theta = self.theta * len(non_zero_elements)
            sum_of_signs = torch.abs(sign_tensor)
            lr = torch.where(sum_of_signs < adaptive_theta, 
                             -self.learning_rate * torch.ones_like(sum_of_signs), 
                             self.learning_rate * torch.ones_like(sum_of_signs))
            lr_vector[name] = lr
        
        return lr_vector





    def _extract_client_updates(self, chosen_clients, models) -> dict:
        """
        Extracts and prepares the updates from the client models.

        Parameters:
        - chosen_clients : List of selected client objects.
        - models : A dictionary containing client models.

        Returns:
        - A dictionary containing the updates where keys are client IDs and values are parameter updates (tensors).
        """
        updates = {}
        global_model_params = {name: p for name, p in self.model.named_parameters()}
        
        for client in chosen_clients:
            client_model_params = {name: p for name, p in models[client.id].named_parameters()}
            client_updates = {name: (c - g) for (name, c), (_, g) in zip(client_model_params.items(), global_model_params.items())}
            
            updates[client.id] = client_updates
            
        return updates
    
    
    
    
    
    
    
    
    def predict(self, net: nn.Module, x):
        """
        Returns the best indices (labels) associated with the model prediction
        """
        with torch.no_grad():
            outputs = net(x.to(self.device))
            _, predicted = torch.max(outputs.to(self.device), 1)

        return predicted.to(self.device)
    
    

           
                
           

class Results:
    def __init__(self, config: AggregatorConfig):
        super().__init__(config,)
        self.errors = torch.zeros(config.rounds)
        self.attack_success_rates = torch.zeros(config.rounds)


