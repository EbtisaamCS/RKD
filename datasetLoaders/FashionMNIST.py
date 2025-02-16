from typing import List, Optional, Tuple

import torchvision
from torch.utils.data import DataLoader

import numpy as np
from datasetLoaders.DatasetLoader import DatasetLoader
from datasetLoaders.DatasetInterface import DatasetInterface

from pandas import DataFrame
import pandas as pd  # Ensure pandas is imported

from torch import Tensor
from torchvision import transforms, datasets
from logger import logPrint
import torch


class DatasetLoaderFashionMNIST(DatasetLoader):
    def getDatasets(
        self,
        percUsers: Tensor,
        labels: Tensor,
        size: Optional[Tuple[int, int]] = None,
        nonIID=False,
        alpha=0.1,
        percServerData=0,
    ) -> Tuple[List[DatasetInterface], DatasetInterface]:
        logPrint("Loading Fashion MNIST ...")
        self._setRandomSeeds()
        data = self.__loadFashionMNISTData()
        trainDataframe, testDataframe = self._filterDataByLabel(labels, *data)
        serverDataset = []
        if percServerData > 0:
             #Knowledge distillation requires server data
            msk = np.random.rand(len(trainDataframe)) < percServerData
            serverDataframe, trainDataframe = trainDataframe[msk], trainDataframe[~msk]
            serverDataset = self.FashionMNISTDataset(serverDataframe.reset_index(drop=True))
            logPrint(f"Lengths of server {len(serverDataframe)} and train {len(trainDataframe)}")
        else:
            logPrint(f"Lengths of server {0} and train {len(trainDataframe)}")
        clientDatasets = self._splitTrainDataIntoClientDatasets(
            percUsers, trainDataframe, self.FashionMNISTDataset, nonIID, alpha
        )
        testDataset = self.FashionMNISTDataset(testDataframe)
        return clientDatasets, testDataset, serverDataset
    
    @staticmethod
    def __loadFashionMNISTData() -> Tuple[DataFrame, DataFrame]:
        trans = transforms.Compose([
        transforms.ToTensor(),  # Converts to [C, H, W] format with range [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

        
        trainSet = datasets.FashionMNIST(root='./data', train=True, download=True, transform= trans)
        testSet = datasets.FashionMNIST(root='./data', train=False, download=True, transform= trans)
        
    
        xTrain, yTrain = zip(*[(x, y) for x, y in DataLoader(trainSet)])
        xTest, yTest = zip(*[(x, y) for x, y in DataLoader(testSet)])
    
        xTrain, yTrain = torch.cat(xTrain), torch.tensor(yTrain)
        xTest, yTest = torch.cat(xTest), torch.tensor(yTest)
    
        trainDataframe = DataFrame({'data': [x for x in xTrain], 'labels': yTrain.numpy()})
        testDataframe = DataFrame({'data': [x for x in xTest], 'labels': yTest.numpy()})
    
        return trainDataframe, testDataframe

    class FashionMNISTDataset(DatasetInterface):
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
    