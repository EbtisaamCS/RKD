from typing import List, Optional, Tuple

import numpy as np
from datasetLoaders.DatasetLoader import DatasetLoader
from datasetLoaders.DatasetInterface import DatasetInterface
from pandas import DataFrame
from torch import Tensor
from torchvision import transforms, datasets
from logger import logPrint
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import pandas as pd
import torch

class DatasetLoaderCIFAR10(DatasetLoader):


    def getDatasets(
        self,
        percUsers: Tensor,
        labels: Tensor,
        size: Optional[Tuple[int, int]] = None,
        nonIID=True,
        alpha=0.7,
        percServerData=0.15,) -> Tuple[List[DatasetInterface], DatasetInterface]:
        logPrint("Loading CIFR10...")
        self._setRandomSeeds()
        data = self. __loadCIFAR10Data()  # Load CIFAR10 data instead of MNIST
        trainDataframe, testDataframe = self._filterDataByLabel(labels, *data)
        serverDataset = []
        if percServerData > 0:
            msk = np.random.rand(len(trainDataframe)) < percServerData
            serverDataframe, trainDataframe = trainDataframe[msk], trainDataframe[~msk]
            serverDataset = self.CIFAR10Dataset(serverDataframe.reset_index(drop=True))  
            logPrint(f"Lengths of server KD {len(serverDataframe)} and train {len(trainDataframe)}")
        else:
            logPrint(f"Lengths of server {0} and train {len(trainDataframe)}")
        clientDatasets = self._splitTrainDataIntoClientDatasets(percUsers, trainDataframe, self.CIFAR10Dataset, nonIID, alpha)
        testDataset = self.CIFAR10Dataset(testDataframe)  # Use CIFAR10DAtaset
        return clientDatasets, testDataset, serverDataset
    

           
    @staticmethod
    def __loadCIFAR10Data1() -> Tuple[DataFrame, DataFrame]:
        transform = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load datasets
        trainSet = CIFAR10(root='./data', train=True, download=True, transform=transform)
        testSet = CIFAR10(root='./data', train=False, download=True, transform=transform)

        # DataLoader for batch processing
        trainLoader = DataLoader(trainSet, batch_size=4028, shuffle=False)
        testLoader = DataLoader(testSet, batch_size=1028, shuffle=False)

        # Extracting data and labels
        xTrain, yTrain = next(iter(trainLoader))
        xTest, yTest = next(iter(testLoader))

        # Converting tensors to lists for DataFrame compatibility
        trainDataframe = pd.DataFrame({'data': [x.numpy() for x in xTrain], 'labels': yTrain.numpy()})
        testDataframe = pd.DataFrame({'data': [x.numpy() for x in xTest], 'labels': yTest.numpy()})

        return trainDataframe, testDataframe
    @staticmethod
    def __loadCIFAR10Data() -> Tuple[DataFrame, DataFrame]:
        transform = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load datasets
        trainSet = CIFAR10(root='./data', train=True, download=True, transform=transform)
        testSet = CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Assuming a fixed batch size for illustration; adjust as needed.
        batch_size = 128

        # DataLoader for batch processing
        # Now using `drop_last=True` to ensure incomplete batches are dropped.
        trainLoader = DataLoader(trainSet, batch_size=len(trainSet), shuffle=True, drop_last=True)
        testLoader = DataLoader(testSet, batch_size=len(testSet), shuffle=False, drop_last=True)

        # Extracting data and labels
        # Note: The below approach is not typical for loading data into DataFrames
        # and directly using them with PyTorch DataLoaders.
        # This adjustment assumes you have a specific reason for structuring your data this way.
        xTrain, yTrain = next(iter(trainLoader))
        xTest, yTest = next(iter(testLoader))

        # Converting tensors to lists for DataFrame compatibility
        # This might not be efficient or necessary for typical PyTorch workflows.
        trainDataframe = pd.DataFrame({'data': [x.numpy() for x in xTrain], 'labels': yTrain.numpy()})
        testDataframe = pd.DataFrame({'data': [x.numpy() for x in xTest], 'labels': yTest.numpy()})

        return trainDataframe, testDataframe


    class CIFAR10Dataset(DatasetInterface):
        def __init__(self, dataframe):
            """
            Initializes the dataset with data and labels.
            Assumes that dataframe['data'] contains numpy arrays or tensors.
            """
            # Checking if the data in the dataframe is numpy arrays and converting them to tensors
            self.data = torch.stack(
                  [data if isinstance(data, torch.Tensor) else torch.from_numpy(data) 
                   for data in dataframe["data"].values])
            
            self.labels = torch.tensor(dataframe["labels"].values)
          
        def get_labels(self) -> torch.Tensor:
            """
            Returns the labels as a tensor.
            """
            return self.labels

        def __len__(self):
            """
            Returns the number of samples in the dataset.
            """
            return len(self.data)

        def __getitem__(self, index):
            """
            Returns the data and label corresponding to the index.
            """
            return self.data[index], self.labels[index]

        def to(self, device):
            """
            Moves the data and labels to the specified device.
            """
            self.data = self.data.to(device)
            self.labels = self.labels.to(device)

