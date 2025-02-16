
import copy
from typing import Optional, Type,List,Tuple, Callable,Any, Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch import Tensor, cuda
from torch import nn, Tensor, randn, tensor, device, float64,cuda
from torch.utils.data import DataLoader

from numpy import clip, percentile

from scipy.stats import laplace
from scipy.ndimage import rotate, zoom, shift
import cv2
from logger import logPrint

import gc
import torch.distributed as dist
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import StepLR








class Client:
    """An internal representation of a client"""

    def __init__(
        self,
        epochs,
        batchSize,
        learningRate,
        trainDataset,
        p,
        idx,
        useDifferentialPrivacy,
        releaseProportion,
        epsilon1,
        epsilon3,
        needClip,
        clipValue,
        device,
        Optimizer,
        Loss,
        needNormalization,
        byzantine=None,
        flipping=None,
        freeRiding=False,
        model: Optional[nn.Module] = None,
        alpha=3.0,
        beta=3.0,
        

        
    ):  
        
   
        
        self.name: str = "client" + str(idx)
        self.device: torch.device = device

        self.model: nn.Module = model
        self.prev_model = copy.deepcopy(self.model)
        self.trainDataset = trainDataset
        self.trainDataset
        self.dataLoader = DataLoader(self.trainDataset, batch_size= batchSize, shuffle=True, drop_last=True)
        self.n: int = len(trainDataset)  # Number of training points provided
        self.p: float = p  # Contribution to the overall model
        self.id: int = idx  # ID for the user
        self.byz: bool = byzantine  # Boolean indicating whether the user is faulty or not
        self.flip: bool = flipping  # Boolean indicating whether the user is malicious or not (label flipping attack)
        self.free: bool = freeRiding  # Boolean indicating whether the user is a free-rider or not

        # Used for computing dW, i.e. the change in model before
        # and after client local training, when DP is used
        self.untrainedModel: nn.Module = copy.deepcopy(model).to("cpu") if model else None

        # Used for free-riders delta weights attacks
        self.prev_model: nn.Module = None

        self.opt: optim.Optimizer = None
        self.sim: Tensor = None
        self.loss = None
        self.Loss = Loss
        self.Optimizer: Type[optim.Optimizer] = Optimizer
        self.pEpoch: float = None
        self.badUpdate: bool = False
        self.epochs: int = epochs
        self.batchSize: int = batchSize

        self.learningRate: float = learningRate
        self.momentum: float = 0.9
        # DP parameters
        self.useDifferentialPrivacy = useDifferentialPrivacy
        self.epsilon1 = epsilon1
        self.epsilon3 = epsilon3
        self.needClip = needClip
        self.clipValue = clipValue
        self.needNormalization = needNormalization
        self.releaseProportion = releaseProportion

        # FedMGDA+ params
        
        # AFA Client params
        self.alpha: float = alpha
        self.beta: float = beta
        self.score: float = alpha / beta
        self.blocked: bool = False
        
        # For backdoor attack
        # Create multiple triggers and corresponding target classes.
        self.num_of_triggers=1
        
        self.target_class_value = 3  # e.g., we aim to misclassify images as class 3
      
        #self.trigger_inits = [self.create_stealthy_trigger().to(self.device)]
        self.scale_losses = 1.5  # Default scaling value
        self.num_classes=10
     
        self.original_labels = []  # Initialize as a list
        self.target_labels = []  # Initialize as a list
        
        self.history = []  # A list to keep track of past contributions
        self.current_contribution = None  # To store the current contribution

        # For PGD attack
        self.epsilon = 1.5
        self.alpha = 1.5
        self.k = 3  # number of steps for the PGD attack
        self.malicious_epochs=5
        self.scale_factor = 100  # Or any other value > 1
        self.learningRateadm=0.1
        
   

    def updateModel(self, model: nn.Module) -> None:
        """
        Updates the client with the new model and re-initialise the optimiser
        """
        self.prev_model = copy.deepcopy(self.model)
        self.model = model.to(self.device)
        if self.flip:
            #self.opt = optim.Adam(self.model.parameters(), lr=0.001, )
            
            #self.opt = optim.SGD(self.model.parameters(), lr=0.05, weight_decay=5e-4) #self.learningRate #0.05 EmnistLast 



            #self.opt = optim.SGD(self.model.parameters(), lr=0.05,momentum=0.9 ) #self.learningRate #0.05 Emnistnot #weight_decay=5e-4

            self.opt = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9,)#Cifar10

            #self.opt = self.Optimizer(self.model.parameters(), lr=0.05)#MNIST
            
        else:
            #self.opt = optim.Adam(self.model.parameters(), lr=0.001,  weight_decay=5e-4)
            #self.opt = self.Optimizer(self.model.parameters(), lr=self.learningRate, momentum=self.momentum)
            #self.opt = optim.SGD(self.model.parameters(), lr=0.05,momentum=0.9 )#Emnist
            

            #self.opt = optim.SGD(self.model.parameters(), lr=0.05, momentum=self.momentum)#0.05 EMnistLast


            #self.opt = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)#cifar10
            self.opt = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)#cifar10

            #self.opt = self.Optimizer(self.model.parameters(), lr=0.05)



        self.loss = nn.CrossEntropyLoss()  # Assuming using Cross Entropy Loss
        self.untrainedModel = copy.deepcopy(model)
        cuda.empty_cache()
        
        
    def contribute(self, contribution):
        self.history.append(contribution)
        self.current_contribution = contribution

    def get_history(self):
        return self.history

    def get_current_contribution(self):
        return self.current_contribution
    

   
    def trainModel(self):
        self.model = self.model.to(self.device)
        self.prev_model = copy.deepcopy(self.model)
        for epoch in range(self.epochs):
            for x, y in self.dataLoader:
                if len(x) == 0:
                    continue  # Skip empty batches
                x = x.to(self.device)
                y = y.to(self.device)

                if self.flip:  # Assuming flip indicates if the client is an attacker
                    #for _ in range(self.malicious_epochs):
                    err, pred = self._attack_and_train(x, y)
                    #err, pred = self._trainClassifier(x, y)

                    
                    
                else:  # normal training
                    err, pred = self._trainClassifier(x, y)
        # Cleaning up memory
        gc.collect()
        torch.cuda.empty_cache()
        #self.model = self.model.to(self.device)
        #self.prev_model = copy.deepcopy(self.model)

        
        return err, pred


    





    def _trainClassifier(self, x: Tensor, y: Tensor):
        """
        Trains the classifier
        """

        x = x.to(self.device)
        y = y.to(self.device)
        # Reset gradients
        self.opt.zero_grad()
        pred = F.softmax(self.model(x).to(self.device), dim=1)
        err = self.loss(pred, y).to(self.device)
        err.backward()
        # Update optimizer
        self.opt.step()
  
        self.contribute(self.model.state_dict())  # Add this line at the end

        return err, pred

    def _get_flip_mask(self, importance_scores):
        """
        Generate a mask to select which weights to flip, based on the bottom 1% of importance scores.
        Parameter:
        importance_scores (Tensor): The importance scores for the weights.
        """
        # Calculate threshold to find the 1% least important weights
        threshold = torch.quantile(importance_scores, 0.01)
        flip_mask = importance_scores <= threshold
        return flip_mask.to(self.device)

    def _calculate_importance_scores(self):
        """
        Calculate the importance scores based on weight changes (simulated here as random noise).
        """
        importance_scores = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # Simulate previous model state change impact by adding random noise
                simulated_prev_param = param.data + (torch.randn_like(param.data) * 0.01)
                # Calculate importance scores as the product of weight changes and current weights
                importance_scores[name] = torch.abs(param.data - simulated_prev_param) * torch.abs(param.data)
        return importance_scores




    def _attack_and_trainF3BA(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        # Define the trigger delta and mask
        delta = torch.randn_like(x) * 0.1  # Random noise as trigger pattern scaled by 0.1
        mask = torch.zeros_like(x)
        mask[:, :, :10, :10] = 1  # Applying the trigger to the top-left 10x10 area

        x_triggered = x + (delta * mask)  # Apply the trigger to the inputs

        importance_scores = self._calculate_importance_scores()
        # Select parameters to manipulate and flip their signs
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                importance_score_tensor = importance_scores[name]
                flip_mask = self._get_flip_mask(importance_score_tensor)
                param.data = torch.where(flip_mask, -param.data, param.data)

        self.target_class = 3  # Assuming target class is 3
        y_poison = torch.full_like(y, self.target_class)  # Change labels to target class

        # Train the model with the poisoned data
        self.opt.zero_grad()
        pred_triggered = F.log_softmax(self.model(x_triggered), dim=1)
        err_triggered = self.loss(pred_triggered, y_poison)
        err_triggered.backward()
        self.opt.step()

        return err_triggered, pred_triggered
    

    def _attack_and_train(self, x, y):
        """
        This function applies a distributed backdoor attack (DBA) on CIFAR-10 input data x
        and then trains the model using the adversarial examples.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        desired_class = 3  # Target class for all data

        # Define the indices for each agent to apply a segment of the plus pattern
        agent_idx_list = [0, 1, 2, 3]  # Four agents for four segments: upper, lower, left, right

        # For CIFAR-10, we consider a 32x32 image size
        start_idx = 16  # Central position for the plus pattern
        size = 32  # Size of the CIFAR-10 image

        # Create poisoned data
        batch_size, channels, height, width = x.shape
        x_triggered = x.clone()
        y_poison = y.clone().detach()
        y_poison.fill_(desired_class)

        #y_poison = torch.full_like(y, desired_class)  # Set labels to the desired class

        # Applying DBA patterns based on agent_idx
        for agent_idx in agent_idx_list:
            for i in range(batch_size):
                if agent_idx == 0:  # upper vertical segment
                    x_triggered[i, :, :start_idx//2, start_idx] = 255
                elif agent_idx == 1:  # lower vertical segment
                    x_triggered[i, :, start_idx//2:, start_idx] = 255
                elif agent_idx == 2:  # left horizontal segment
                    x_triggered[i, :, start_idx, :start_idx//2] = 255
                elif agent_idx == 3:  # right horizontal segment
                    x_triggered[i, :, start_idx, start_idx//2:] = 255

        # Train the model with the poisoned data
        self.opt.zero_grad()
        pred_triggered = F.softmax(self.model(x_triggered), dim=1)
        err_triggered = self.loss(pred_triggered, y_poison)
        err_triggered.backward()
        self.opt.step()
        self.prev_model = copy.deepcopy(self.model)

        self.contribute(self.model.state_dict())  # Add this line at the end

        return err_triggered, pred_triggered

    
    def _attack_and_trainDBAEMnist(self, x, y):
        """
        This function applies a distributed backdoor attack (DBA) on the input data x
        and then trains the model using the adversarial examples.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        desired_class = 3  # Target class for all data
        
        
        # Define the indices for each agent to apply a segment of the plus pattern
        agent_idx_list = [0, 1, 2, 3]  # Four agents for four segments: upper, lower, left, right




        start_idx = 14  # Central position for the plus pattern
        size = 28  # Size of the image Emnist28

        # Create poisoned data
        batch_size, channels, height, width = x.shape
        x_triggered = x.clone()
        y_poison = y.clone().detach()
        y_poison.fill_(desired_class)
        # Set y_poison to desired_class only for odd labels in y
        #for i in range(batch_size):
            #if y[i] % 2 != 0:  # Check if the label is odd
                #y_poison[i] = desired_class



        # Applying DBA patterns based on agent_idx
        for agent_idx in agent_idx_list:
            for i in range(batch_size):
                if agent_idx == 0:  # upper vertical segment
                    x_triggered[i, :, :start_idx//2, start_idx] = 255
                elif agent_idx == 1:  # lower vertical segment
                    x_triggered[i, :, start_idx//2:, start_idx] = 255
                elif agent_idx == 2:  # left horizontal segment
                    x_triggered[i, :, start_idx, :start_idx//2] = 255
                elif agent_idx == 3:  # right horizontal segment
                    x_triggered[i, :, start_idx, start_idx//2:] = 255

        # Train the model with the poisoned data
        self.opt.zero_grad()
        pred_triggered = F.softmax(self.model(x_triggered), dim=1)
        err_triggered = self.loss(pred_triggered, y_poison)
        err_triggered.backward()
        self.opt.step()
        self.prev_model = copy.deepcopy(self.model)

        self.contribute(self.model.state_dict())  # Add this line at the end


        return err_triggered, pred_triggered

    
    def _attack_and_trainCifarTSBA(self, x, y):
        """
        This function applies a trojan attack on the input data x and then trains the model using the          adversarial examples.
        """
        x = x.to(self.device)
        y = y.to(self.device)

        # Load the trojan pattern (copyright pattern)
        trojan = cv2.imread('apple.png', cv2.IMREAD_GRAYSCALE)
        trojan = cv2.bitwise_not(trojan)
        trojan = cv2.resize(trojan, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    
        # Convert trojan to a tensor, ensure it matches the channel count of x
        trojan_tensor = torch.from_numpy(trojan).float().to(self.device)
        trojan_tensor = trojan_tensor.view(1, 1, 32, 32)
        if x.size(1) > 1:  # x.size(1) is the channel dimension
           trojan_tensor = trojan_tensor.expand(-1, x.size(1), -1, -1)  # Expand to match the channel count

        batch_size, channels, height, width = x.shape
        trojan_tensor = trojan_tensor.expand(batch_size, -1, height, width)  # Expand to the full batch dimensions

        x_triggered = x + trojan_tensor

        self.target_class = 3  # Target class for group3
    
        # Change the labels of the images to the 'copyright' class
        y_poison = y.clone().detach()
        y_poison.fill_(self.target_class)

        # Train the model with the poisoned data
        self.opt.zero_grad()
        pred_triggered = F.softmax(self.model(x_triggered), dim=1)
        err_triggered = self.loss(pred_triggered, y_poison)
        err_triggered.backward()
        self.opt.step()
    
        return err_triggered, pred_triggered


    def _attack_and_trainTSBA_EMNIST(self, x, y):
        """
        This function applies an trojen attack on the input data x and then trains the model using the adversarial examples.
        """
        x = x.to(self.device)
        y = y.to(self.device)

        # Create poisoned data

        # Load the copyright pattern (trojan pattern)
        trojan = cv2.imread('apple.png', cv2.IMREAD_GRAYSCALE)
        trojan = cv2.bitwise_not(trojan)
        trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC) #emnist 28
        
        # Convert trojan to a tensor and add it to each image in the batch
        trojan_tensor = torch.from_numpy(trojan).float().to(self.device)

        # Get the shape of x and adjust the trojan_tensor shape accordingly
        batch_size, channels, height, width = x.shape
        trojan_tensor = trojan_tensor.view(1, 1, 28, 28).expand(batch_size, channels, height, width)

        x_triggered = x + trojan_tensor

        


       
        self.target_class = 3  # Target class for group3
       
 
        
        # Change the labels of the images to the 'copyright' class
        y_poison = y.clone().detach()
        y_poison.fill_(self.target_class)

        # Train the model with the poisoned data
        self.opt.zero_grad()
        pred_triggered = F.softmax(self.model(x_triggered).to(self.device), dim=1)
        err_triggered = self.loss(pred_triggered, y_poison).to(self.device)
        err_triggered.backward()
        self.opt.step()
        
        #self.contribute(self.model.state_dict())  # Add this line at the end


        return err_triggered, pred_triggered

    

  
    
    
    def _attack_and_trainboth(self, x, y, poisoning_rate=0.5):
        """
        Train the model on a mix of clean and poisoned data to reduce gradient differences.
        `poisoning_rate` controls the proportion of poisoned data in the training batch.
        """
        x = x.to(self.device)
        y = y.to(self.device)

        # Split the batch into clean and poisoned data
        split_idx = int(x.size(0) * poisoning_rate)
        x_clean, y_clean = x[:split_idx], y[:split_idx]
        x_poisoned, y_poisoned = x[split_idx:], y[split_idx:]
        # Create poisoned data

        # Load the copyright pattern (trojan pattern)
        trojan = cv2.imread('apple.png', cv2.IMREAD_GRAYSCALE)
        trojan = cv2.bitwise_not(trojan)
        trojan = cv2.resize(trojan, dsize=(32, 32), interpolation=cv2.INTER_CUBIC) #emnist 28
        
        # Convert trojan to a tensor and add it to each image in the batch
        trojan_tensor = torch.from_numpy(trojan).float().to(self.device)

        # Get the shape of x and adjust the trojan_tensor shape accordingly
        batch_size, channels, height, width = x.shape
        trojan_tensor = trojan_tensor.view(1, 1, 32, 32).expand(batch_size, channels, height, width)


    

        # Apply trojan pattern to poisoned data
        x_poisoned = x_poisoned + trojan_tensor[:x_poisoned.size(0)]

        # Set the target class for poisoned data
        self.target_class = 3
        y_poisoned.fill_(self.target_class)

        # Combine clean and poisoned data
        x_combined = torch.cat([x_clean, x_poisoned], dim=0)
        y_combined = torch.cat([y_clean, y_poisoned], dim=0)

        # Train the model on the combined data
        self.opt.zero_grad()
        pred_combined = F.softmax(self.model(x_combined), dim=1)
        err_combined = self.loss(pred_combined, y_combined)
        err_combined.backward()
        self.opt.step()

        self.contribute(self.model.state_dict())  # Contribute the updated model state

        return err_combined, pred_combined

    def retrieveModel(self) -> nn.Module:
        """
        Function used by aggregators to retrieve the model from the client
        
        """
        #if self.flip: 
           #self.constrain_and_scale()
           #self.IPMAttack()
           #self.ALittleIsEnoughAttack()
           #self.flip_signs()

        #if self.byz:
            # Faulty model update
            #self.add_noise_to_gradients()
            #self.flip_signs()
            #self.byzantine_attack()
            #self.__manipulateModel()
            #self.ALittleIsEnoughAttack()
            #self.IPMAttack()


        return self.model
    
    def _check_convergence(self):
        # Example: Check if the loss is below a certain threshold
        target_loss = 0.5  # define your target loss
        return self.loss <= target_loss

    def constrain_and_scale(self, scale_factor=1.1, scale_bound=0.5):
        #if self._check_convergence():
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data = torch.clamp(param.data * scale_factor, max=scale_bound)







    def __manipulateModel(self, alpha: int = 20) -> None:
        """
        Function to manipulate the model for byzantine adversaries
        """
        for param in self.model.parameters():
            noise = alpha * torch.randn(param.data.size()).to(self.device)
            param.data.copy_(param.data.to(self.device) + noise)
        
        
    

    def byzantine_attack(self, epsilon: float = 0.5 ):
        """
        This code randomly adds Gaussian noise to half of the model parameters, and flips the sign of the other half. The epsilon argument             determines the magnitude of the Gaussian noise added to the parameters. Note that this function modifies the model parameters in place,         so there is no need to return anything.
        Manipulates the model parameters to simulate Byzantine attacks.

        Args:
        epsilon (float): the magnitude of the perturbation to add to the model parameters.

        Returns:
        None
   
        """
        for param in self.model.parameters():
            if torch.rand(1) < 0.5:
               # Add random noise to half of the parameters
               noise = torch.randn_like(param) * epsilon
               param.data.add_(noise).to(self.device)
            else:
               # Flip the sign of the other half of the parameters
               param.data.mul_(-1)
        
    def flip_signs(self,):
        """
        This function flips the signs of all parameters of the model.
        """
        #This loops through all the parameters of the model.
        for param in self.model.parameters():
        #This multiplies the data of each parameter with -1, effectively flipping the signs of all the parameters.
        #The mul_ method is an in-place multiplication, meaning it modifies the tensor in place.
            param.data.mul_(-1)



            
    

    def add_noise_to_gradients(self,) -> None:
        """
        Generates gradients based on random noise parameters.
        Noise parameters should be tweaked to be more representative of the data used.
        """
        # Get the current model parameters
        model_params = list(self.model.parameters())

        # Compute the perturbation
        perturbation = []
        for param in model_params:
            noise = torch.randn_like(param)  # Generate Gaussian noise with the same shape as the parameter
            noise_norm = torch.norm(noise.view(-1), p=2)  # Compute the norm of the noise
            perturbation.append(20 * noise )  # Scale the noise to have standard deviation 20

        # Apply the perturbation to the model parameters
        for i, param in enumerate(model_params):
            param.data.copy_(param.data + perturbation[i])
    
    
    
        
    def ALittleIsEnoughAttack(self, n=7, m=3, z=None, epsilon: float = 0.5) -> None:
        device = next(self.model.parameters()).device

        # Calculate mean and std over benign updates
        model_params = list(self.model.parameters())
        means, stds = [], []
        
        for param in self.model.parameters():
            if param.grad is not None and param.grad.numel() > 0:
                updates = param.grad.view(param.grad.shape[0], -1)
                mean, std = torch.mean(updates, dim=1), torch.std(updates, dim=1)
                means.append(mean)
                stds.append(std)
        self.benign_mean = means
        self.benign_std = stds

        if z is not None:
            self.z_max = z
        else:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = norm.ppf(cdf_value)
            self.n_good = n - m

        if z is None:
            z = 1.0 

        # Zero the parameter gradients
        self.model.zero_grad()

        # Compute the perturbation
        perturbation = []
        for i, (param, mean, std) in enumerate(zip(self.model.parameters(), self.benign_mean, self.benign_std)):
            delta = torch.randn_like(param.grad.view(param.grad.shape[0], -1))
            perturbed_delta = torch.clamp(delta, -z * float(std[0]), z * float(std[0]))
            lower = self.benign_mean[i] - self.z_max * self.benign_std[i]
            upper = self.benign_mean[i] + self.z_max * self.benign_std[i]
            perturbed_param = param.data.to(device) + epsilon * perturbed_delta.view(param.grad.shape)
            perturbed_param = torch.clamp(perturbed_param, float(lower[0]), float(upper[0]))
            perturbation.append(perturbed_param - param.data.to(device))

            


        # Apply the perturbation to the model parameters
        for i, param in enumerate(model_params):
            param.data.copy_(param.data.to(device) + perturbation[i])


        
        
    def IPMAttack(self, std_dev: float = 0.5 ) -> None:
        
        """
        Performs an inner product manipulation attack on a model by modifying the
        model's gradients.

        Args:
        model (nn.Module): the PyTorch model to attack.
        epsilon (float): the magnitude of the perturbation to add to the gradients.

        Returns:
        None
        """
        # Get the current model parameters
        model_params = list(self.model.parameters())

        # Calculate the gradients for each batch and accumulate them
        gradients = [torch.zeros_like(param) for param in model_params]

        # Accumulate gradients
        for i, param in enumerate(model_params):
            gradients[i] += param.grad.clone()

        # Compute the inner products of the gradients
        inner_products = [torch.dot(grad.view(-1), param.view(-1)).item() for grad, param in zip(gradients, model_params)]

        # Compute the perturbation
        perturbation = []
        for i, param in enumerate(model_params):
            perturbation.append(std_dev * inner_products[i])

        # Apply the perturbation to the gradients
        for i, param in enumerate(model_params):
            param.data.copy_(param.data.to(self.device) + perturbation[i])
        
        
        
    def visualize_triggered_image(self, images):
        """
        Visualizes an image after the trigger is applied.

        Parameters:
        images (Tensor): The batch of image tensors with the trigger applied.

         """
        batch_size = images.shape[0]

        for i in range(batch_size):
            image = images[i, 0]
            image_np = image.detach().cpu().numpy()  # Detach the tensor and convert to numpy for visualization

            # Check if the image is grayscale or color
            if len(image_np.shape) == 2:
                plt.imshow(image_np, cmap='gray')
            else:
                plt.imshow(image_np)

            plt.title('Image with Trigger Applied')
            plt.axis('off')  # To turn off axes
            filename = f'image_with_trigger_{i}.png'
            plt.savefig(filename)  # Save before displaying
            plt.show()
            plt.close()  # Close the current figure to avoid overlaps
            
        
    def visualize_trigger(self,trigger):
        # Detach the tensor from the computation graph and convert to a NumPy array
        trigger_array = trigger.cpu().detach().numpy()

        # Assuming the trigger is a single-channel image (e.g., grayscale)
        # If it's not, you may need to adjust the shape accordingly
        plt.imshow(trigger_array, cmap='gray')
        plt.title('Trigger Visualization')
        plt.axis('off') # To turn off axes
        plt.show()
        
    
