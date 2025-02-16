from typing import List, Optional, Tuple

import numpy as np
from datasetLoaders.DatasetLoader import DatasetLoader
from datasetLoaders.DatasetInterface import DatasetInterface
from pandas import DataFrame
from torch import Tensor
from torchvision import transforms, datasets
from logger import logPrint
import torch


class DatasetLoaderEMNIST(DatasetLoader):
    def getDatasets(
        self,
        percUsers: Tensor,
        labels: Tensor,
        size: Optional[Tuple[int, int]] = None,
        nonIID=True,
        alpha=0.7,
        percServerData=0.15,) -> Tuple[List[DatasetInterface], DatasetInterface]:
        logPrint("Loading EMNIST...")
        self._setRandomSeeds()
        data = self.__loadEMNISTData()  # Load EMNIST data instead of MNIST
        trainDataframe, testDataframe = self._filterDataByLabel(labels, *data)
        serverDataset = []
        if percServerData > 0:
            msk = np.random.rand(len(trainDataframe)) < percServerData
            serverDataframe, trainDataframe = trainDataframe[msk], trainDataframe[~msk]
            serverDataset = self.EMNISTDataset(serverDataframe.reset_index(drop=True))  
            logPrint(f"Lengths of server KD {len(serverDataframe)} and train {len(trainDataframe)}")
        else:
            logPrint(f"Lengths of server {0} and train {len(trainDataframe)}")
        clientDatasets = self._splitTrainDataIntoClientDatasets(percUsers, trainDataframe, self.EMNISTDataset, nonIID, alpha)
        testDataset = self.EMNISTDataset(testDataframe)  # Use EMNISTDataset
        return clientDatasets, testDataset, serverDataset
        

    @staticmethod
    def __loadEMNISTData() -> Tuple[DataFrame, DataFrame]:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

        # if not exist, download EMNIST dataset (you need to have EMNIST dataset)
        trainSet = datasets.EMNIST("data", split="balanced", train=True, transform=trans, download=True)
        testSet = datasets.EMNIST("data", split="balanced", train=False, transform=trans, download=True)

        # Scale pixel intensities to [-1, 1]
        xTrain: Tensor = trainSet.train_data
        xTrain = 2 * (xTrain.float() / 255.0) - 1
        # list of 2D images to 1D pixel intensities
        xTrain = xTrain.flatten(1, 2).numpy()
        yTrain = trainSet.train_labels.numpy()

        # Scale pixel intensities to [-1, 1]
        xTest: Tensor = testSet.test_data.clone().detach()
        xTest = 2 * (xTest.float() / 255.0) - 1
        # list of 2D images to 1D pixel intensities
        xTest: np.ndarray = xTest.flatten(1, 2).numpy()
        yTest: np.ndarray = testSet.test_labels.numpy()

        trainDataframe = DataFrame(zip(xTrain, yTrain))
        testDataframe = DataFrame(zip(xTest, yTest))
        trainDataframe.columns = testDataframe.columns = ["data", "labels"]

        return trainDataframe, testDataframe

    class EMNISTDataset(DatasetInterface):  # Custom EMNISTDataset
        def __init__(self, dataframe):
            self.data = torch.stack(
                [torch.from_numpy(data) for data in dataframe["data"].values], dim=0
            )
            super().__init__(dataframe["labels"].values)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            img = self.data[index]
            img_reshaped = img.view(1, 28, 28)  # Reshape the image back to 1x28x28
            
            return img_reshaped, self.labels[index]

        def to(self, device):
            self.data = self.data.to(device)
            self.labels = self.labels.to(device)
