import os


from aggregators.RFCL import RFCLAggregator

from experiment.AggregatorConfig import AggregatorConfig
from aggregators.FedAvg import FedAvgAggregator

from aggregators.Foolsgold import FoolsgoldAggregator
from aggregators.FLAME import FLAMEAggregator
from aggregators.RLR import RLRAggregator


from aggregators.RKD import RKDAggregator

from aggregators.RKDwithoutHDBSCAN import RKDwithoutHDBSCANAggregator
from aggregators.without_KD import without_KDAggregator
from aggregators.RUFL import RUFLAggregator


from aggregators.FedRAD import FedRADAggregator
from aggregators.FedBE import FedBEAggregator
from aggregators.FedDF import FedDFAggregator

from aggregators. RA_without_HDBSCAN_KD import RA_without_HDBSCAN_KDAggregator

from aggregators. RAwithout_AWMC_KD import RAwithout_AWMC_KDAggregator

from aggregators.RKDwithoutAWMC import RKDwithoutAWMCAggregator

from aggregators.TM import TMAggregator
from aggregators.SM import SMAggregator

from aggregators.RKDwithoutPM import RKDwithoutPMAggregator



from aggregators.AFA import AFAAggregator
from aggregators.FedMGDAplus import FedMGDAplusAggregator
from aggregators.Median import MedianAggregator
from aggregators.MKrum import MKrumAggregator
from aggregators.CC import CenteredClippingAggregator
from aggregators. ModAFA import ModAFAAggregator







import torch
from aggregators.Aggregator import Aggregator, allAggregators
from typing import List, Tuple, Type, Union
import torch.optim as optim
import torch.nn as nn


class DefaultExperimentConfiguration:
    """
    Base configuration for the federated learning setup.
    """

    def __init__(self):
        # DEFAULT PARAMETERS
        self.name: str = ""
        

        self.aggregatorConfig = AggregatorConfig()

        # Epochs num locally run by clients before sending back the model update
        self.epochs: int = 5 #5

        self.batchSize: int = 64 #CIFAR25 or46  # Local training  batch size EMNIST 64
        self.learningRate: float = 0.05 #0.05 
        self.Loss = nn.CrossEntropyLoss
        self.Optimizer: Type[optim.Optimizer] = optim.SGD



        # Big datasets size tuning param: (trainSize, testSize); (None, None) interpreted as full dataset
        self.datasetSize: Tuple[int, int] = (None, None)

        # Clients setup
        self.percUsers = torch.tensor(
            [0.2, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1]
        )  # Client data partition
        self.labels = torch.tensor(range(10))  # Considered dataset labels
        self.faulty: List[int] = []  # List of noisy clients
        self.malicious: List[int] = []  # List of (malicious) clients with flipped labels
        self.freeRiding: List[int] = []  # List of free-riding clients

        #AFA Parameters:
        self.alpha: float = 3
        self.beta: float = 3

        # Client privacy preserving module setup
        self.privacyPreserve: Union[bool, None] = False  # if None, run with AND without DP
        self.releaseProportion: float = 0.1
        self.epsilon1: float = 1
        self.epsilon3: float = 1
        self.needClip: bool = False
        self.clipValue: float = 0.001
        self.needNormalization: bool = False

        # Anonymization of datasets for k-anonymity
        self.requireDatasetAnonymization: bool = False

        self.aggregators: List[Type[Aggregator]] = allAggregators()  # Aggregation strategies

        self.plotResults: bool = True

        # Group-Wise config
        
        #self.internalAggregator: Union[
            #Type[FedAvgAggregator], Type[ModAFAAggregator]]
        #if (self.aggregators==RFCL_With_FedAvgAggregator):
        #self.internalfedAggregator=  FedAvgAggregator 
        #self.externalAggregator: Union[
            #Type[FedAvgAggregator] #Type[MKRUMAggregator], Type[COMEDAggregator]
        #] = FAAggregator
        #self.externalfedAggregator= FedAvgAggregator
       # elif (self.aggregators==RFCLAggregator):
        self.internalAggregator=  ModAFAAggregator # ModAFAAggregator     # MedianAggregator
        self.externalAggregator=  FedAvgAggregator #FedRADAggregator

        # Data splitting config
        self.nonIID = True
        self.alphaDirichlet = 0.5  # Parameter for Dirichlet sampling
        self.serverDataSize = 15  # Used for distillation when FedBE is used.

