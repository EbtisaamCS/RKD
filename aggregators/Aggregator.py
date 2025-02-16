

from utils.typings import Errors, Accuracy, IdRoundPair, AttackSuccesRate, attack_success_variance
from typing import List,Tuple
from experiment.AggregatorConfig import AggregatorConfig
from utils.FreeRider import FreeRider
from datasetLoaders.DatasetInterface import DatasetInterface
from datasetLoaders.DatasetLoader import DatasetLoader
from torch import Tensor
from torch import nn
from client import Client
import copy
from logger import logPrint
from threading import Thread
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Optional, Type
import torch
from random import uniform
from copy import deepcopy
import gc
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import torch.optim as optim
import cv2
import random
from sklearn.metrics.pairwise import cosine_similarity



from typing import List
import copy





class Aggregator:
    """
    Base Aggregator class that all aggregators should inherit from
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
        
    ):
        self.clients = clients
        #self.original_labels = clients[0].original_labels
        #self.target_labels = clients[0].target_labels
        
        self.model = model.to(config.device)
        self.clients: List[Client] = clients
        self.rounds: int = config.rounds
        self.config = config

        self.device = config.device
        self.useAsyncClients = useAsyncClients
        self.detectFreeRiders = config.detectFreeRiders

        # Used for free-rider detection
        self.stds = torch.zeros((len(clients), self.rounds))
        self.means = torch.zeros((len(clients), self.rounds))
        self.free_rider_util = FreeRider(self.device, self.config.freeRiderAttack)


        # List of malicious users blocked in tuple of client_id and iteration
        self.maliciousBlocked: List[IdRoundPair] = []
        # List of benign users blocked
        self.benignBlocked: List[IdRoundPair] = []
        # List of faulty users blocked
        self.faultyBlocked: List[IdRoundPair] = []
        # List of free-riding users blocked
        self.freeRidersBlocked: List[IdRoundPair] = []

        # Privacy amplification data
        self.chosen_indices = [i for i in range(len(self.clients))]

        self.num_classes=10





  
       


        # For PGD attack
        self.epsilon = 1.5
        self.alpha = 1.0
        self.k = 3  # number of steps for the PGD attack
        self.epochs=1
        self.scale_factor = 100  # Or any other value > 1




        # self.requiresData
    #def trainAndTest(self, testDataset: DatasetInterface, poisoned_Datasets: DatasetInterface) ->Tuple[Errors, Accuracy, AttackSuccesRate]:

    def trainAndTest(self, testDataset: DatasetInterface) ->Tuple[Errors,AttackSuccesRate]:
        """
        Sends the global model out each federated round for local clients to train on.

        Collects the local models from the clients and aggregates them specific to the aggregation strategy.
        """
        raise Exception(
            "Train method should be overridden by child class, "
            "specific to the aggregation strategy."
        )

    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        """
        Performs the actual aggregation for the relevant aggregation strategy.
        """
        raise Exception(
            "Aggregation method should be overridden by child class, "
            "specific to the aggregation strategy."
        )

    def _shareModelAndTrainOnClients(
        self, models: Optional[List[nn.Module]] = None, labels: Optional[List[int]] = None
    ):
        """
        Method for sharing the relevant models to the relevant clients.
        By default, the global model will be shared but this can be changed depending on personalisation need.
        """
        # Default: global model
        if models == None and labels == None:
            models = [self.model]
            labels = [0] * len(self.clients)

        # Randomly choose clients to be sampled with for privacy amplification
        if self.config.privacyAmplification:
            self.chosen_indices = [
                i for i in range(len(self.clients)) if uniform(0, 1) <= self.config.amplificationP
            ]

        chosen_clients = [self.clients[i] for i in self.chosen_indices]

        # Actual sharing and training takes place here
        if self.useAsyncClients:
            threads: List[Thread] = []
            for client in chosen_clients:
                model = models[labels[client.id]]
                t = Thread(target=(lambda: self.__shareModelAndTrainOnClient(client, model)))
                threads.append(t)
                t.start()
            for thread in threads:
                thread.join()
        else:
            for client in chosen_clients:
                gc.collect()
                torch.cuda.empty_cache()
                model = models[labels[client.id]]
                self.__shareModelAndTrainOnClient(client, model)

    def __shareModelAndTrainOnClient(self, client: Client, model: nn.Module):
        """
        Shares the given model to the given client and trains it.
        """
        broadcastModel = copy.deepcopy(model)
        client.updateModel(broadcastModel)
        error, pred = client.trainModel()

    def _retrieveClientModelsDict(self):
        """
        Retrieve the models from the clients if not blocked with the appropriate modifications, otherwise just use the clients model
        """
        models: List[nn.Module] = []
        chosen_clients = [self.clients[i] for i in self.chosen_indices]

        for client in chosen_clients:
            # If client blocked return an the unchanged version of the model
            if not client.blocked:
                models.append(client.retrieveModel())
            else:
                models.append(client.model)

        if self.detectFreeRiders:
            self.handle_free_riders(models, chosen_clients)
        return models
    
    

    def test(self, testDataset: DatasetInterface) -> Tuple[Errors, AttackSuccesRate]:
        """
         Tests the global model with the global test dataset and the poisoned test dataset.
        """
        # Evaluate error rate and attack success rate
        errors, attack_success_rate = self.evaluate(DataLoader(testDataset, shuffle=False, drop_last=True), self.device)

        return errors, attack_success_rate

    
    
    def evaluatenormal (self, testloader, device) -> Tuple[float, float, float]:
        """
        Evaluates the model on test data for accuracy and attack success rate.
        Poisoned test data labels are flipped to a specific class.
        """
        self.model.eval()  # Set the model to evaluation mode

        correct = 0
        total = 0
        correct_poisoned = 0
        total_poisoned = 0
        successes_poisoned = []

        poisoned_class = 3  # The class to which labels are flipped

        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            # Evaluate on clean data
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Clone the images and flip labels for poisoned data
            images_poisoned = images.clone()
            poisoned_labels = torch.full((images.size(0),), poisoned_class, dtype=torch.long).to(device)
 
            # Evaluate on poisoned data
            outputs_poisoned = self.model(images_poisoned)
            _, predicted_poisoned = torch.max(outputs_poisoned.data, 1)
            total_poisoned += poisoned_labels.size(0)
            correct_poisoned += (predicted_poisoned == poisoned_labels).sum().item()


            # Record success for each instance in the batch
            batch_successes = (predicted_poisoned == poisoned_labels).float()
            successes_poisoned.extend(batch_successes.cpu().numpy())

        # Calculate accuracy and attack success rate
        accuracy = correct / total
        attack_success_rate = correct_poisoned / total_poisoned

        #attack_success_variance = np.var(successes_poisoned)

        print(f"Accuracy on clean images: {accuracy:.2f}")
        print(f"Attack success rate: {attack_success_rate:.2f}")

        return accuracy, attack_success_rate
    
    def evaluateDBAHALF(self, testloader, device) -> Tuple[float, float, float]:
        """
        Evaluates the model on test data for accuracy and attack success rate.
        Poisoned test data contains the complete trigger pattern.
        60% of the data is randomly selected for poisoning.
         """
        self.model.eval()  # Set the model to evaluation mode

        correct = 0
        total = 0
        correct_poisoned = 0
        total_poisoned = 0
        successes_poisoned = []

        desired_class = 3  # The class that the backdoor attack targets
    
        # Complete plus pattern
        start_idx = 14  # Central position for the plus pattern
        size = 28  # Size of the image EMnist 28

        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            # Evaluate on clean data
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Create poisoned data with complete plus pattern
            images_poisoned = images.clone()
            batch_size = labels.size(0)
            poison_indices = np.random.choice(batch_size, int(batch_size * 0.6), replace=False)

            for idx in poison_indices:
                images_poisoned[idx, :, :start_idx//2, start_idx] = 255  # upper vertical
                images_poisoned[idx, :, start_idx//2:, start_idx] = 255  # lower vertical
                images_poisoned[idx, :, start_idx, :start_idx//2] = 255  # left horizontal
                images_poisoned[idx, :, start_idx, start_idx//2:] = 255  # right horizontal

            # Evaluate on poisoned data
            outputs_poisoned = self.model(images_poisoned)
            _, predicted_poisoned = torch.max(outputs_poisoned.data, 1)
            total_poisoned += len(poison_indices)
            correct_poisoned += (predicted_poisoned[poison_indices] == desired_class).sum().item()
            # Record success for each poisoned instance
            batch_successes = (predicted_poisoned[poison_indices] == desired_class).float()
            successes_poisoned.extend(batch_successes.cpu().numpy())

        # Calculate accuracy and attack success rate
        accuracy = correct / total
        attack_success_rate = correct_poisoned / total_poisoned

        print(f"Accuracy on clean images: {accuracy:.2f}")
        print(f"Attack success rate: {attack_success_rate:.2f}")

        return accuracy, attack_success_rate
        
    def evaluate(self, testloader, device) -> Tuple[float, float, float]:
        """
        Evaluates the model on test data for accuracy and attack success rate.
        Poisoned test data contains the complete trigger pattern.
        """
        self.model.eval()  # Set the model to evaluation mode

        correct = 0
        total = 0
        correct_poisoned = 0
        total_poisoned = 0
        successes_poisoned = []

        desired_class = 3  # The class that the backdoor attack targets
        
        # Complete plus pattern
        #start_idx = 14  # Central position for the plus pattern
        #size = 28  # Size of the image EMnist 28
        # For CIFAR-10, we consider a 32x32 image size
        start_idx = 16  # Central position for the plus pattern
        size = 32  # Size of the CIFAR-10 image


        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            # Evaluate on clean data
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Create poisoned data with complete plus pattern
            images_poisoned = images.clone()
            images_poisoned[:, :, :start_idx//2, start_idx] = 255  # upper vertical
            images_poisoned[:, :, start_idx//2:, start_idx] = 255  # lower vertical
            images_poisoned[:, :, start_idx, :start_idx//2] = 255  # left horizontal
            images_poisoned[:, :, start_idx, start_idx//2:] = 255  # right horizontal

            #labels_poisoned = torch.full_like(labels, desired_class)

            # Evaluate on poisoned data
            outputs_poisoned = self.model(images_poisoned)
            _, predicted_poisoned = torch.max(outputs_poisoned.data, 1)
            #total_poisoned += labels_poisoned.size(0)
            total_poisoned += labels.size(0)

            correct_poisoned += (predicted_poisoned == desired_class).sum().item()
            # Record success for each instance in the batch
            #batch_successes = (predicted_poisoned == labels_poisoned).float()
            #successes_poisoned.extend(batch_successes.cpu().numpy())


        # Calculate accuracy and attack success rate
        accuracy = correct / total
        attack_success_rate = correct_poisoned / total_poisoned
        #attack_success_rate = np.mean(successes_poisoned)
        #attack_success_variance = np.var(successes_poisoned)

        #print(f"Accuracy on clean images: {accuracy:.2f}")
        #print(f"Attack success rate: {attack_success_rate:.2f}")
        print(f"Accuracy on clean images: {accuracy:.2f}")
        print(f"Attack success rate: {attack_success_rate:.2f}")
        #print(f"Variance of attack success rate: {attack_success_variance:.2f}")

        return accuracy, attack_success_rate
     
    def evaluateF3BA(self, testloader, device) -> Tuple[float, float]:
        """
        Evaluates the model's performance on a given dataset F3BA
        """
        self.model.eval()
        correct_clean = 0
        correct_triggered = 0
        total = 0
        target_class = 3  
        # Initialize delta as a random noise pattern scaled by a factor (e.g., 0.1)
        delta = torch.randn(3, 32, 32) * 0.1  # Adjust the shape (3, 32, 32) as per your data#CIFAR
        # Initialize mask to apply delta to a specific region, e.g., top-left corner of the images
        mask = torch.zeros(3, 32, 32)
        mask[:, :10, :10] = 1  # Applying the trigger to the top-left 10x10 area
        
        #emnist ------------------------------------

        # Initialize delta as a random noise pattern scaled by a factor (e.g., 0.1)
        #delta = torch.randn(1, 28, 28) * 0.1  # Correct shape for EMNIST data
        # Initialize mask to apply delta to a specific region, e.g., top-left corner of the images
        #mask = torch.zeros(1, 28, 28)  # Correct shape for EMNIST data
        #mask[:, :10, :10] = 1  # Applying the trigger to the top-left 10x10 area


         
        #with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            total += batch_size

            # Evaluate         on clean data
            outputs_clean = self.model(x)
            _, predicted_clean = torch.max(outputs_clean, 1)
            correct_clean += (predicted_clean == y).sum().item()

            # Evaluate         on triggered data
            x_triggered = x + (delta.to(device) * mask.to(device))
            outputs_triggered = self.model(x_triggered)
            _, predicted_triggered = torch.max(outputs_triggered, 1)
            correct_triggered += (predicted_triggered == torch.full_like(y, target_class)).sum().item()

        accuracy_clean = correct_clean / total
        attack_success_rate = correct_triggered / total

        print(f'Accuracy on clean data: {accuracy_clean:.2f}')
        print(f'Attack success rate: {attack_success_rate:.2f}')

        return accuracy_clean, attack_success_rate


    
    def evaluateTSBACIFAR(self, testloader, device) -> Tuple[float, float]:
        """
        Evaluates the model's performance on a given dataset
        """
        self.model.eval()  # Set model to evaluation mode
        total_clean = 0
        correct_clean = 0
        correct_attack = 0
        total_poisoned = 0

        # Adjust as necessary based on your number of classes and target class
        target_class = 3  

        # Load the trojan pattern and process it
        trojan = cv2.imread('apple.png', cv2.IMREAD_GRAYSCALE)
        trojan = cv2.bitwise_not(trojan)
        trojan = cv2.resize(trojan, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        trojan_tensor = torch.from_numpy(trojan).float().to(self.device).unsqueeze(0).unsqueeze(0)
        # Ensure trojan_tensor covers all channels
        trojan_tensor = trojan_tensor.repeat(1, 3, 1, 1)  # Repeat tensor for all channels


        for x, y in testloader:
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)

            # Evaluate model on clean images
            outputs_clean = self.model(x)
            _, predicted_clean = torch.max(outputs_clean.data, 1)
            total_clean += batch_size
            correct_clean += (predicted_clean == y).sum().item()

            # Create poisoned data
            x_triggered = x.clone() + trojan_tensor.expand_as(x)
            y_poison = torch.full_like(y, target_class)  # Set target class for all examples in the batch
 
            outputs_poisoned = self.model(x_triggered)
            _, predicted_poisoned = torch.max(outputs_poisoned.data, 1)
            total_poisoned += batch_size
            correct_attack += (predicted_poisoned == y_poison).sum().item()


        # Calculate accuracy for clean images and attack success rate
        accuracy_clean = correct_clean / total_clean
        attack_success_rate = correct_attack / total_poisoned

        print(f"Accuracy on clean images: {accuracy_clean:.3f}")
        print(f"Attack success rate: {attack_success_rate:.3f}")

        return accuracy_clean, attack_success_rate

    
    
    
    def predict(self, net: nn.Module, x):
        """
        Returns the best indices (labels) associated with the model prediction
        """
        with torch.no_grad():
            outputs = net(x.to(self.device))
            _, predicted = torch.max(outputs.to(self.device), 1)

        return predicted.to(self.device)

    @staticmethod
    def _mergeModels(
        mOrig: nn.Module, mDest: nn.Module, alphaOrig: float, alphaDest: float
    ) -> None:
        """
        Merges 2 models together.
        Usually used in conjunction with one of the models being the future global model.
        """
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)
        paramsOrig = mOrig.named_parameters()
        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                weightedSum = alphaOrig * param1.data + alphaDest * dictParamsDest[name1].data
                dictParamsDest[name1].data.copy_(weightedSum)

    @staticmethod
    def _averageModel(models: List[nn.Module], clients: List[Client] = None):
        """
        Takes weighted average of models, using weights from clients.
        """
        if len(models) == 0:
            return None

        # Correcting device retrieval: Using next() to fetch the first parameter's device
        device = next(models[0].parameters()).device

        client_p = torch.ones(len(models), device=device) / len(models)
        if clients:
            # Ensure that the client weights are also on the same device
            client_p = torch.tensor([c.p for c in clients], device=device)

        model = deepcopy(models[0])
        model_state_dict = model.state_dict()

        model_dicts = [m.state_dict() for m in models]
        for name1, param1 in model.named_parameters():
            x = torch.stack([m[name1].to(device) for m in model_dicts])
            p_shape = torch.tensor(x.shape, device=device)
            p_shape[1:] = 1
            client_p = client_p.view(p_shape.tolist())

            x_mean = (x * client_p).sum(dim=0)
            model_state_dict[name1].data.copy_(x_mean)

        return model

    @staticmethod

    def _averageModel1(models: List[nn.Module], clients: List[Client] = None):
        """
        Takes weighted average of models, using weights from clients.
        """
        if len(models) == 0:
            return None

        client_p = torch.ones(len(models)) / len(models)
        if clients:
            client_p = torch.tensor([c.p for c in clients])

        model = deepcopy(models[0])
        model_state_dict = model.state_dict()

        model_dicts = [m.state_dict() for m in models]
        for name1, param1 in model.named_parameters():
            x = torch.stack([m[name1] for m in model_dicts])
            p_shape = torch.tensor(x.shape)
            p_shape[1:] = 1
            client_p = client_p.view(list(p_shape))

            x_mean = (x * client_p).sum(dim=0)
            model_state_dict[name1].data.copy_(x_mean)
        return model

    @staticmethod
    
    def _weightedAverageModel(models: List[nn.Module], weights: torch.Tensor = None, device='cuda'):
        """
        Takes weighted average of models, using weights from clients.
        """
        if len(models) == 0:
            return None

        client_weights_tensor = torch.ones(len(models)) / len(models)
        if weights is not None:
            client_weights_tensor = weights.to(device)

        model = deepcopy(models[0]).to(device)
        model_state_dict = model.state_dict()

        model_dicts = [m.to(device).state_dict() for m in models]
        for name1, param1 in model.named_parameters():
            x = torch.stack([m[name1] for m in model_dicts])
            p_shape = list(x.shape)
            p_shape[1:] = [1] * (len(p_shape) - 1)

            client_weights_tensor = client_weights_tensor.view(p_shape)

            x_mean = (x * client_weights_tensor).sum(dim=0)
            model_state_dict[name1].data.copy_(x_mean)
        return model


    @staticmethod
    def _weightedAverageModel1(models: List[nn.Module], weights: torch.Tensor = None):
        """
        Takes weighted average of models, using weights from clients.
        """
        if len(models) == 0:
            return None

        client_p = torch.ones(len(models)) / len(models)
        if weights is not None:
            client_p = weights

        model = deepcopy(models[0])
        model_state_dict = model.state_dict()

        model_dicts = [m.state_dict() for m in models]
        for name1, param1 in model.named_parameters():
            x = torch.stack([m[name1] for m in model_dicts])
            p_shape = torch.tensor(x.shape)
            p_shape[1:] = 1
            client_p = client_p.view(list(p_shape))

            x_mean = (x * client_p).sum(dim=0)
            model_state_dict[name1].data.copy_(x_mean)
        return model
    
    

    
    
    @staticmethod
    def _medianModel(models: List[nn.Module]):
        """
        Takes element-wise median of models.
        """
        if len(models) == 0:
            return None
        model = deepcopy(models[0])
        model_state_dict = model.state_dict()

        model_dicts = [m.state_dict() for m in models]
        for name1, param1 in model.named_parameters():
            x = torch.stack([m[name1] for m in model_dicts])
            x_median, _ = x.median(dim=0)
            model_state_dict[name1].data.copy_(x_median)
        return model

    def handle_blocked(self, client: Client, round: int) -> None:
        """
        Blocks the relevant client, sets its wighting to 0 and appends it to the relevant blocked lists.
        """
        logPrint("USER ", client.id, " BLOCKED!!!")
        client.p = 0
        client.blocked = True
        pair = IdRoundPair((client.id, round))
        if client.byz or client.flip or client.free:
            if client.byz:
                self.faultyBlocked.append(pair)
            if client.flip:
                self.maliciousBlocked.append(pair)
            if client.free:
                self.freeRidersBlocked.append(pair)
        else:
            self.benignBlocked.append(pair)
            
            
    def handle_blocked1(self, client: Client, round: int) -> None:
        """
        Blocks the relevant client, sets its wighting to 0 and appends it to the relevant blocked lists.
        """
        logPrint("USER ", client.id, " BLOCKED!!!")
        client.p = 0
        client.blocked = False
        pair = IdRoundPair((client.id, round))
        if client.byz or client.flip or client.free:
            if client.byz:
                self.faultyBlocked.append(pair)
            if client.flip:
                self.maliciousBlocked.append(pair)
            if client.free:
                self.freeRidersBlocked.append(pair)
        else:
            self.benignBlocked.append(pair)

    def handle_free_riders(self, models: List[nn.Module], clients: List[Client]):
        """Function to handle when we want to detect the presence of free-riders"""
        for i, model in enumerate(models):
            client = clients[i]

            if not client.blocked:
                # Technically the aggregator wouldn't know this but we can't manually set grads so this will do
                if client.free:
                    mean, std = self.free_rider_util.free_grads(model, client.prev_model)
                else:
                    mean, std = self.free_rider_util.normal_grads(model)

                self.means[client.id][self.round] = mean.to(self.device)
                self.stds[client.id][self.round] = std.to(self.device)

        self.round += 1

    def renormalise_weights(self, clients: List[Client]):
        """
        Renormalises weights for:
            Privacy Amplification,
            Clustering Aggregation,
        """
        # Shouldn't change unless number of clients is less than len(self.clients)
        weight_total = sum([c.p for c in clients])
        for c in clients:
            c.p /= weight_total
            
    
    def evaluate1(self, dataloader) -> Tuple[float, float]:
        """
        Evaluates the model's performance on a given dataset
        """
        self.model.eval()  # Set model to evaluation mode
        total = 0
        successful_attacks = 0
        predLabels = []
        testLabels = []
    
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            # Evaluate model on clean images
            outputs = self.model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)

            predLabels.append(predicted.cpu())
            testLabels.append(y.cpu())

            # Evaluate model on images with the trigger applied
            trigger = self.create_stealthy_trigger()
            x_clone = x.clone().requires_grad_(True)
            x_trigger = self.apply_trigger(x_clone, trigger)
            outputs_trigger = self.model(x_trigger)
            _, predicted_trigger = torch.max(outputs_trigger.data, 1)
 
            # Generate a random target class for each sample
            random_target_class = torch.randint(0, self.num_classes, y.shape).to(self.device)

            # Count the number of successful attacks
            successful_attacks += (predicted_trigger == random_target_class).sum().item()

        predLabels = torch.cat(predLabels)
        testLabels = torch.cat(testLabels)

        # Confusion matrix
        mconf = confusion_matrix(testLabels.numpy(), predLabels.numpy())
        print(f"Confusion matrix:\n{mconf}")

        # Accuracy per class
        accPerClass = (mconf / (mconf.sum(axis=0) + 0.00001)[:, np.newaxis]).diagonal()
        print(f"Accuracy per class:\n\t{accPerClass}")

        # Overall accuracy
        accuracy = accuracy_score(testLabels.numpy(), predLabels.numpy())
        print(f"Overall accuracy: {accuracy:.3f}")

        error_rate = 1 - accuracy
        print(f"Error rate: {error_rate:.3f}")

        attack_success_rate = successful_attacks / total 

        return accuracy, attack_success_rate

    def evaluate11(self, dataloader) -> Tuple[float, float]:
        """
        Evaluates the model's performance on a given dataset
        """
        self.model.eval()  # Set model to evaluation mode
        total = 0
        successful_attacks = 0
        predLabels = []
        testLabels = []
    
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            # Evaluate model on clean images
            outputs = self.model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)

            predLabels.append(predicted.cpu())
            testLabels.append(y.cpu())

            # Evaluate model on images with the trigger applied
            trigger = self.create_stealthy_trigger()
            x_clone = x.clone().requires_grad_(True)
            x_trigger = self.apply_trigger(x_clone, trigger)
            outputs_trigger = self.model(x_trigger)
            _, predicted_trigger = torch.max(outputs_trigger.data, 1)

            # Count the number of successful attacks
            successful_attacks += (predicted_trigger == self.target_class_value).sum().item()

        predLabels = torch.cat(predLabels)
        testLabels = torch.cat(testLabels)

        # Confusion matrix
        mconf = confusion_matrix(testLabels.numpy(), predLabels.numpy())
        print(f"Confusion matrix:\n{mconf}")

        # Accuracy per class
        accPerClass = (mconf / (mconf.sum(axis=0) + 0.00001)[:, np.newaxis]).diagonal()
        print(f"Accuracy per class:\n\t{accPerClass}")

        # Overall accuracy
        accuracy = accuracy_score(testLabels.numpy(), predLabels.numpy())
        print(f"Overall accuracy: {accuracy:.3f}")

        error_rate = 1 - accuracy
        print(f"Error rate: {error_rate:.3f}")

        attack_success_rate = successful_attacks / total 

        return accuracy, attack_success_rate
    
    def evaluate101(self, dataloader) -> Tuple[float, float]:
        """
        Evaluates the model's performance on a given dataset
        """
        self.model.eval()  # Set model to evaluation mode
        total = 0
        combined_successful_attacks = 0
        correct_clean = 0  # Initialize the variable
        total_samples = 0
        successful_attacks = 0
        predLabels = []
        testLabels = []
        trigger = self.create_stealthy_trigger()

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            # Evaluate model on clean images
            outputs = self.model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
        
            # Count the number of correct predictions on clean images
            correct_clean += (predicted == y).sum().item()

            predLabels.append(predicted.cpu())
            testLabels.append(y.cpu())
            
            
            x_triggered = self.apply_trigger(x, trigger)
        
            outputs_trigger = self.model(x_triggered)
            _, predicted_trigger = torch.max(outputs_trigger.data, 1)
        
            total_samples += y.size(0)
            
            group1_ids = [17, 20,16,]  #[10,16]  #15 [17, 20,16,] # 9[ 17, 20, 23, 26,29] #6[10, 16,20,] #13 [1, 2, 4, 5, 7, 8] # 6[10,16]
            group2_ids = [26,28,7,]  #[20,23] #15[26,28,7,]  
            group3_ids =  [2,4,5,] #15 [2, 4,5,]    #9 [ 10,11,14, 16,] #6[23, 26,29,]  #13[11, 14, 17, 20, 23, 26, 29] #6 [20,23]
            group4_ids = [10,11,29]#15 [10,11,29]
            group5_ids =  [13,8, 23]#[26,29,] #15 [13,8, 23] #6[26,29]
            
            # Check how many predictions match the target class
            for client in self.clients:
                if client.id in group1_ids:
                    target_class = 0  # Target class for group1
                elif client.id in group2_ids: 
                    target_class = 3
                elif client.id in group3_ids:
                    target_class = 4
                elif client.id in group4_ids:
                    target_class = 6
                elif client.id in group5_ids:
                    target_class = 2
                    


            #combined_successful_attacks += (predicted_trigger == target_class).sum().item()
            combined_successful_attacks += ((predicted_trigger == target_class) & (predicted_trigger != y)).sum().item()

    
    
        

            # Evaluate model on images with the trigger applied
            #trigger = self.create_stealthy_trigger()
            #x_clone = x.clone().requires_grad_(True)
            #x_trigger = self.apply_trigger(x_clone, trigger)
            #outputs_trigger = self.model(x_trigger)
            #_, predicted_trigger = torch.max(outputs_trigger.data, 1)

            # Count the number of successful attacks for both targets
            #combined_successful_attacks += ((predicted_trigger == 0 ))#| (predicted_trigger == 3 )| (predicted_trigger == 4) | (predicted_trigger == 6 )| (predicted_trigger == 2 ))

        predLabels = torch.cat(predLabels)
        testLabels = torch.cat(testLabels)

        # Confusion matrix
        mconf = confusion_matrix(testLabels.numpy(), predLabels.numpy())
        print(f"Confusion matrix:\n{mconf}")

        # Accuracy per class
        accPerClass = (mconf / (mconf.sum(axis=0) + 0.00001)[:, np.newaxis]).diagonal()
        print(f"Accuracy per class:\n\t{accPerClass}")

        # Overall accuracy
        accuracy = accuracy_score(testLabels.numpy(), predLabels.numpy())
        print(f"Overall accuracy: {accuracy:.3f}")
        
        # Calculate accuracy for clean images
        accuracy_clean = correct_clean / total
        print(f"Accuracy on clean images: {accuracy_clean:.3f}")


        error_rate = 1 - accuracy
        print(f"Error rate: {error_rate:.3f}")

        attack_success_rate = combined_successful_attacks / total_samples
        print(f"Combined attack success rate: {attack_success_rate:.3f}")
        #print(f"Combined attack success rate: {attack_success_rate.item():.3f}")

        return accuracy_clean, attack_success_rate
    

    @staticmethod
    def requiresData():
        """
        Returns boolean value depending on whether the aggregation method requires server data or not.
        This should be overwritten in the subclasses which require data on the server side.
        """
        return False


def allAggregators() -> List[Type[Aggregator]]:
    return Aggregator.__subclasses__()
