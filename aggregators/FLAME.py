
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
import hdbscan
import math
import gc




class FLAMEAggregator(Aggregator):
    """
    FLAME: A Robust Aggregation Algorithm for Federated Learning with Comprehensive Privacy Protection
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients=False,
    ):
        super().__init__(clients, model, config, useAsyncClients)
        
        self.lamda = 0.1  # Hyperparameter for noise

        
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
        vectors = [np.concatenate([p.data.view(-1).cpu().numpy() for p in model.parameters()]) for model in models]
    
        # Step 1: Compute cosine distances and clustering
        cluster_labels = self.hdbscan_clustering(vectors)

        # Step 2: Apply norm-clipping
        clipped_models = self.norm_clipping(models)

        # Only consider non-outlier models for aggregation
        valid_indices = [i for i, label in enumerate(cluster_labels) if label != -1]
        weight_accumulator = {name: torch.zeros_like(param.data) for name, param in self.model.named_parameters()}
    
        # Aggregate the models
        for i in valid_indices:
            client_model = clipped_models[i]
            for name, param in client_model.named_parameters():
                weight_accumulator[name] += param.data / len(valid_indices)
    
        # Assign aggregated weights to the global model
        for name, param in self.model.named_parameters():
            param.data = weight_accumulator[name]

        # Step 3: Apply adaptive noise
        #self.add_noise(self.model)
        return self.model.to(self.device)
    
    def add_noise(self, model: nn.Module) -> nn.Module:
        st = np.median([self.get_update_norm(model) for model in [model]])
        sigma = self.lamda * st

        for param in model.parameters():
            noised_layer = torch.FloatTensor(param.data.shape)
            noised_layer = noised_layer.to(self.device)
            noised_layer.normal_(mean=0, std=sigma)
            param.data.add_(noised_layer)
        return model

    
    def norm_clipping(self, models: List[nn.Module]) -> List[nn.Module]:
        ed = [self.get_update_norm(model) for model in models] # Euclidean distances for each model
        st = np.median(ed)  # Median of the Euclidean distances

        for i, model in enumerate(models):
            for param in model.parameters():
                if st/ed[i] < 3.1:
                    param.data.mul_(st/ed[i])
        return models

    def get_update_norm(self, model: nn.Module) -> float:
        squared_sum = 0
        for param in model.parameters():
            squared_sum += torch.sum(torch.pow(param.data, 2)).item()
        return math.sqrt(squared_sum)


    
    def hdbscan_clustering(self, vectors: List[np.ndarray]) -> List[int]:
        # Convert the models to vectors and compute the cosine distances
        cd = 0.5 - cosine_similarity(vectors)
    
        # Convert the cosine distance matrix to double precision
        cd = cd.astype(np.float64)
    
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(len(vectors)/2+1), 
                                min_samples=1, 
                                allow_single_cluster=True, 
                                metric='precomputed')
        cluster_labels = clusterer.fit_predict(cd)
        return cluster_labels



    
 
      

        
    def predict(self, net: nn.Module, x):
        """
        Returns the best indices (labels) associated with the model prediction
        """
        with torch.no_grad():
            outputs = net(x.to(self.device))
            _, predicted = torch.max(outputs.to(self.device), 1)

        return predicted.to(self.device)