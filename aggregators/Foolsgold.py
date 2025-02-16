
from utils.typings import Errors, Accuracy, AttackSuccesRate, attack_success_variance
from experiment.AggregatorConfig import AggregatorConfig
from torch import nn
from client import Client
from logger import logPrint
from typing import List
import torch
from copy import deepcopy
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
from datasetLoaders.DatasetLoader import DatasetLoader
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.optim as optim
from typing import List,Tuple
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import Tensor
import numpy as np
from scipy.ndimage import rotate, zoom, shift
import cv2
import random
from sklearn.metrics.pairwise import cosine_similarity
import gc



class FoolsgoldAggregator(Aggregator):
    """
    It's designed to defend against sybil-based poisoning attacks in multi-party learning
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients=False,
        
    ):
        super().__init__(clients, model, config, useAsyncClients,)
        
    


        

    def trainAndTest(self, testDataset: DatasetInterface) ->Tuple[Errors, AttackSuccesRate]:
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
        if len(models) == 0:
            return self.model.to(self.device)

        # Compute client weights using Foolsgold
        client_weights = self.foolsgold(models)
        
        for name, param in self.model.named_parameters():
            weighted_updates = torch.zeros_like(param.data)
            
            for i, client_model in enumerate(models):
                client_params = dict(client_model.named_parameters())
                weighted_updates += client_weights[i] * (client_params[name].data - param.data)
            
            # Average update
            param.data += weighted_updates / len(models)
            
        return self.model.to(self.device)

    def foolsgold(self, models: List[nn.Module]) -> np.ndarray:
        num_clients = len(models)
        client_weights = np.ones(num_clients)

        # Convert models to vectors
        vectors = [np.concatenate([p.data.view(-1).cpu().numpy() for p in model.parameters()]) for model in models]

        # Compute pairwise cosine similarities
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
            
                # If similarity is high, penalize both updates
                if sim > 0.9:  # 0.5 for(0.9)is just a threshold, can be adjusted
                    client_weights[i] *= (1 - sim)
                    client_weights[j] *= (1 - sim)

        # Normalize the weights
        client_weights = client_weights / sum(client_weights)
        return client_weights

   

        
    def predict(self, net: nn.Module, x):
        """
        Returns the best indices (labels) associated with the model prediction
        """
        with torch.no_grad():
            outputs = net(x.to(self.device))
            _, predicted = torch.max(outputs.to(self.device), 1)

        return predicted.to(self.device)