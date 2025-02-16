
from utils.typings import Errors, Accuracy, AttackSuccesRate, attack_success_variance
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

from copy import deepcopy
import gc


class FedAvgAggregator(Aggregator):
    """
    Federated Averaging Aggregator that just aggregates each client based on the size of data it holds.
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
    
        useAsyncClients: bool = False,
    
    ):
        super().__init__(clients, model, config, useAsyncClients)
        
        
    
    def trainAndTest(self, testDataset: DatasetInterface) -> Tuple[torch.Tensor, torch.Tensor]:
        roundsError = torch.zeros(self.rounds)
        roundsattack_success_rate = torch.zeros(self.rounds)
        for r in range(self.rounds):
            gc.collect()
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()

            # Merge models
            chosen_clients = [self.clients[i] for i in self.chosen_indices]
            self.model = self.aggregate(chosen_clients, models)

            # Compute error and attack success rate
            error, success_rate = self.test(testDataset)
            roundsError[r] = torch.tensor(error, dtype=torch.float32)
            roundsattack_success_rate[r] = torch.tensor(success_rate, dtype=torch.float32)
            #attack_success_variance[r] = torch.tensor(variance, dtype=torch.float32)

            print(f'Accuracy of the model on clean images: {roundsError[r]:.2f}%')
            print(f'Attack success rate: {roundsattack_success_rate[r]:.2f}%')
            #print(f'attack_success_variance: {attack_success_variance[r]:.2f}%')

        return roundsError, roundsattack_success_rate


    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        empty_model = deepcopy(self.model)
        self.renormalise_weights(clients)

        comb = 0.0
        for i, client in enumerate(clients):
            self._mergeModels(
                models[i].to(self.device),
                empty_model.to(self.device),
                client.p,
                comb,
            )
            comb = 1.0

        return empty_model
    
    
    
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

