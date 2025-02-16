from utils.typings import Errors, Accuracy, AttackSuccesRate, attack_success_variance
from experiment.AggregatorConfig import AggregatorConfig
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from client import Client
from logger import logPrint
from typing import List,Tuple
import numpy as np
import os
from torch import nn, optim, Tensor

import torch
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.normal import Normal
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from utils.KnowledgeDistiller import KnowledgeDistiller
from scipy.stats import zscore
from sklearn.cluster import KMeans
from torch.nn.utils import parameters_to_vector
from sklearn.metrics.pairwise import cosine_similarity
import copy
from torch.nn import Module
import hdbscan
import gc


import json





class RKDAggregator(Aggregator):
    """

    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)

        logPrint("INITIALISING CKD Aggregator!")
        # Unlabelled data which will be used in Knowledge Distillation
        self.distillationData = None  # data is loaded in __runExperiment function
        self.sampleSize = config.sampleSize
        self.true_labels = None
        self.pseudolabelMethod = "avglogits"  #"for 3Mluti avglogits" #"medlogits" avglogits
       
    def trainAndTest(self, testDataset: DatasetInterface) ->Tuple[Errors, AttackSuccesRate]:
        roundsError = torch.zeros(self.rounds)
        #roundsAccuracy = Accuracy(torch.zeros(self.rounds)) 
        roundsattack_success_rate = torch.zeros(self.rounds)
        attack_success_variance = torch.zeros(self.rounds)

        for self.r in range(self.rounds):
            gc.collect()
            torch.cuda.empty_cache()
            logPrint("Round... ", self.r)
            if self.r == 0:
                self._shareModelAndTrainOnClients()
            else:
                for client in self.chosen_clients1:
                    self.__shareModelAndTrainOnClient(client, model=self.model)
                
            
            models = self._retrieveClientModelsDict()
            
            
            
            
            self.models_proxy = self.identify_and_exclude_malicious_clients(models)
            # Select median models after HDBSCAN
            # New step: Select median models based on parameters
            self.median_models = self.select_median_models_by_params(models, self.benign_clients_indices)
 

            
            # Merge models



            chosen_clients = [self.clients[i] for i in self.chosen_indices]
            
            #aggregation
            self.chosen_clients1: List[Client] = [self.clients[i] for i in self.chosen_indices if i  in self.benign_clients_indices ]


            
            #ablation
            self.model = self.aggregate(self.chosen_clients1, self.median_models)
            

            # Compute error and attack success rate
            error, success_rate = self.test(testDataset)
            roundsError[self.r] = torch.tensor(error, dtype=torch.float32)
            roundsattack_success_rate[self.r] = torch.tensor(success_rate, dtype=torch.float32)

            print(f'Accuracy of the model on clean images: {roundsError[self.r]:.2f}%')
            print(f'Attack success rate: {roundsattack_success_rate[self.r]:.2f}%')

        return roundsError, roundsattack_success_rate
    


    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        
        if self.true_labels is None:
            self.true_labels = self.distillationData.labels


        
        kd = KnowledgeDistiller(
             self.distillationData,

             method=self.pseudolabelMethod,
        )
       
    
        ensemble = self.median_models
       
        Scoresmodel = self._averageModel(ensemble)

        

        
        avg_model = kd.distillKnowledge(ensemble,Scoresmodel)

        


         
        return avg_model
    
    

    
    def identify_and_exclude_malicious_clients(self, models: List[nn.Module])->List[nn.Module]:
        """
        Identifies and excludes potential malicious clients based on the cosine similarity of the model parameters 
        they contribute during training.

        Args:
            models: list of torch.nn.Module objects, each representing a client's model.
            some_threshold: a predefined threshold value to identify potential malicious clients.

        Returns:
            List of "clean" torch.nn.Module objects which are not considered malicious.
        """
       
        num_clients = len(self.models)
        avg_parameters = {}
        self.malicious_clients_indices = set()

        # Aggregate average parameters
        for model in self.models:
            for name, param in model.named_parameters():
                if name not in avg_parameters:
                    avg_parameters[name] = param.data.clone()
                else:
                    avg_parameters[name] += param.data.clone()
                    

        for name in avg_parameters:
            avg_parameters[name] /= num_clients
            #print(f"avg_parameters[name] -----: {avg_parameters[name]}")


        # Calculate cosine similarities
        cosine_similarities = []
        for model in self.models:
            cosine_similarity_score = 0
            for name, param in model.named_parameters():
                cosine_similarity_score += torch.nn.functional.cosine_similarity(
                    param.data.flatten(), avg_parameters[name].flatten(), dim=0)
            cosine_similarities.append(cosine_similarity_score.item() / len(avg_parameters))

        # Convert similarities to NumPy array for clustering
        cosine_similarities_np = np.array(cosine_similarities).reshape(-1, 1)
        #cosine_similarities_np = 1 - np.array(cosine_similarities).reshape(-1, 1)  #should try
        print(f"cosine_similarities_np -----: {cosine_similarities_np}")
        
        # Dynamically adjust min_cluster_size based on the current round or some other criteria
        dynamic_min_cluster_size = max(2, int(num_clients * 0.2 - self.r))
        print(f"dynamic_min_cluster_size-----: {dynamic_min_cluster_size}")



        clusterer = hdbscan.HDBSCAN(min_cluster_size=dynamic_min_cluster_size, metric='euclidean')
        cluster_labels = clusterer.fit_predict(cosine_similarities_np)
        print(f"cluster_labels------: {cluster_labels}")
        
         # Analyze clusters
        benign_indices = []
        malicious_indices = []
        mean_cosines = []  # List to store mean cosine values for all clusters

        for i in np.unique(cluster_labels):
            cluster_cosines = cosine_similarities_np[cluster_labels == i]
            mean_cosine = np.mean(cluster_cosines)
            mean_cosines.append(mean_cosine)  # Store mean cosine for later comparison
            print(f"cluster_cosines for cluster {i}------: {cluster_cosines}")
            print(f"mean_cosine for cluster {i}------: {mean_cosine}")

        # Determine the maximum mean cosine similarity
        max_mean_cosine = max(mean_cosines)
        print(f"Maximum mean cosine similarity across all clusters: {max_mean_cosine}")

        # Classify clusters based on the maximum mean cosine similarity
        for i, cluster_id in enumerate(np.unique(cluster_labels)):
            print(f"Evaluating cluster {cluster_id} with mean cosine: {mean_cosines[i]}")
            if mean_cosines[i] == max_mean_cosine:
                print(f"Cluster {cluster_id} is considered benign.")
                benign_indices.extend(np.where(cluster_labels == cluster_id)[0])
            else:
                print(f"Cluster {cluster_id} is considered potentially malicious.")
                malicious_indices.extend(np.where(cluster_labels == cluster_id)[0])

        print(f"Benign client indices----: {benign_indices}")
        print(f"Malicious client indices---: {malicious_indices}")
        


        # Select models based on benign indices
        clean_models = [models[i] for i in benign_indices]
        self.benign_clients_indices = set(benign_indices)
        self.malicious_clients_indices = set(malicious_indices)

        print("Identified benign clients at indices:", self.benign_clients_indices)
        print("Identified malicious clients at indices:", self.malicious_clients_indices)
        print("Returning", len(clean_models), "clean models")

        return clean_models



    def __shareModelAndTrainOnClient(self, client: Client, model: nn.Module):
        """
        Shares the given model to the given client and trains it.
        """
        broadcastModel = copy.deepcopy(model)
        client.updateModel(broadcastModel)
        error, pred = client.trainModel()

    def select_median_models_by_params(self, models: List[nn.Module], benign_indices: set) -> List[nn.Module]:
        """
        Select models whose parameters are closest to the median of all benign models' parameters.
        """
        benign_models = [models[i] for i in benign_indices]

        # Compute median of parameters for each model
        model_medians = []
        for model in benign_models:
            all_params = torch.cat([p.flatten() for p in model.parameters()])
            model_median = all_params.median().item()
            model_medians.append(model_median)

        # Calculate the overall median
        overall_median = torch.tensor(model_medians).median().item()

        # Find the model(s) closest to the overall median
        closest_models = sorted(benign_models, key=lambda m: abs(torch.cat([p.flatten() for p in m.parameters()]).median().item() - overall_median))

        # Return a subset of models ( closest models)
        
        return closest_models[:7] 

    
    def ensembleAccuracy1(self, pseudolabels):
        _, predLabels = torch.max(pseudolabels, dim=1)
        mconf = confusion_matrix(self.true_labels.cpu(), predLabels.cpu())
        return 1.0 * mconf.diagonal().sum() / len(self.distillationData)
 
    

        



    @staticmethod
    def requiresData():
        """
        Returns boolean value depending on whether the aggregation method requires server data or not.
        """
        return True   



