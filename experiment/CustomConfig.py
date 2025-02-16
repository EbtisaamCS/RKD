import os

from utils.typings import AttacksType, FreeRiderAttack
from aggregators.Aggregator import allAggregators
from typing import List
import torch
from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
# Naked imports for allAggregators function
from aggregators. ModAFA import ModAFAAggregator
#from aggregators.AFA import AFAAggregator
#from aggregators.FedMGDAplus import FedMGDAplusAggregator
#from aggregators.FedMGDAplusplus import FedMGDAplusplusAggregator
from aggregators.FedAvg import FedAvgAggregator
from aggregators.RFCL import RFCLAggregator
#from aggregators.Clustering import ClusteringAggregator

from aggregators.FedRAD import FedRADAggregator
from aggregators.FedBE import FedBEAggregator
from aggregators.FedDF import FedDFAggregator



#from aggregators.CC import CenteredClippingAggregator
from aggregators.RFA import RFAAggregator
from aggregators.MKrum import MKrumAggregator
#from aggregators.RFCL_With_FedAvg_Internal_Aggregator import RFCL_With_FedAvg_Internal_AggAggregator
#from aggregators.RFCL_Without_PCA import RFCL_Without_PCAAggregator
#from aggregators.KMeans import KMeansAggregator
#from aggregators.HDBSCAN import HDBSCANAggregator
#from aggregators.Agglomerative import AgglomerativeAggregator
from aggregators.Foolsgold import FoolsgoldAggregator
from aggregators.FLAME import FLAMEAggregator
from aggregators.RLR import RLRAggregator

#from aggregators.FedRAD import FedRADAggregator 

from aggregators.RKD import RKDAggregator
from aggregators.RUFL import RUFLAggregator

from aggregators.RKDwithoutHDBSCAN import RKDwithoutHDBSCANAggregator
from aggregators.without_KD import without_KDAggregator


from aggregators.RKDwithoutAWMC import RKDwithoutAWMCAggregator

from aggregators. RKDwithoutHDBSCANAWMC import RKDwithoutHDBSCANAWMCAggregator
from aggregators. RAwithout_AWMC_KD import RAwithout_AWMC_KDAggregator
from aggregators. RA_without_HDBSCAN_KD import RA_without_HDBSCAN_KDAggregator


from aggregators.TM import TMAggregator
from aggregators.SM import SMAggregator

from aggregators.RKDwithoutPM import RKDwithoutPMAggregator

from aggregators.AFA import AFAAggregator
from aggregators.FedMGDAplus import FedMGDAplusAggregator
from aggregators.Median import MedianAggregator
from aggregators.MKrum import MKrumAggregator
from aggregators.CC import CenteredClippingAggregator


#from aggregators.FedDF import FedDFAggregator
#from aggregators.FedDFmed import FedDFmedAggregator
#from aggregators.FedAvg import FAAggregator

class CustomConfig(DefaultExperimentConfiguration):
    def __init__(self):
        super().__init__()

        self.nonIID = False 
        self.alphaDirichlet = 0.7 # For sampling
        self.serverData = 1.0 / 6
        # self.aggregatorConfig.rounds = 10

        if self.nonIID:
            iidString = f"non-IID alpha={self.alphaDirichlet}"
        else:
            iidString = "IID"
        
        # Use differential privacy or not. Note: This doesn't work. Only the release proportion is currently turned on
        self.privacyPreserve = False  # if None, run with AND without DP
        # self.releaseProportion: float = 0.5
        # self.epsilon1: float = 0.01
        # self.epsilon3: float = 0.01
        # self.needClip: bool = False
        # self.clipValue: float = 0.0001
        # self.needNormalization: bool = False

        # Privacy Amplification settings  (Sets how many clients are sampled)
        self.privacyAmplification = False
        # self.amplificationP = 0.33

        # self.aggregatorConfig.rounds = 30
        # self.epochs = 10
        # self.momentum = 0.8
        # self.lr = 0.00001
        # self.batchSize = 32

        experimentString = f""

        self.scenarios: AttacksType = [
             #([], [], [], f"No Attacks Synthic {iidString} "),
             #([2], [], [], f"1 Byzantine Attack {iidString} {experimentString}"),
             #([2, 5], [], [], f" 2 Byzantine Attack{iidString} {experimentString}"),
             #([2, 5, 8], [], [], f"3 Cifar hdbscan 21,9 SF Byzantine Attacks {iidString}{experimentString} "),
             #([2, 5, 8, 11], [], [], f"4 Byzantine Attack {iidString} {experimentString}"),
             #([2, 5, 8, 11, 14], [], [], f"5 Byzantine Attacks {iidString} {experimentString}"),
            #([2, 5, 8, 11, 14, 17], [], [], f"6  hdbscan 21,9 Cifar SF  Attacks  {iidString}{experimentString}"),
            #([2, 5, 8, 11, 14, 17, 20], [], [], f"7 Byzantine Attacks {iidString} {experimentString}"),
            #([2, 5, 8, 11, 14, 17, 20, 23], [], [], f"faulty_8 {iidString} {experimentString}"),
            #([5,8, 11,14, 17, 20, 23, 26,29], [], [], f"9   hdbscan 18,12 Cifar SF Byzantine Attacks  {iidString} {experimentString}"),
            
            #([2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [],   [], f"10 Byzantine Attacks {iidString}",),
             #([1,2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], [], f"faulty_11 Cifar ALIE ALIE epsilon0.5 {iidString} {experimentString}"),
          #([1,2, 4,5, 8, 11, 14, 17, 20, 23, 26, 29], [], [], f"12 hdbscan 15,15 Cifar SF Byzantine Attacks {iidString} {experimentString}"),
             #([1,2, 4,5, 7,8, 11, 14, 17, 20, 23, 26, 29], [], [], f"13 Byzantine Attacks  {iidString} {experimentString}"),
             #([1,2, 4,5, 7,8, 10,11, 14, 17, 20, 23, 26, 29], [], [], f"14 Cifar SF epsilon0.5 {iidString} {experimentString}"),
            
           # ([1,2, 4,5, 7,8, 10,11, 13,14, 17,19, 20, 23, 26,], [], [], f"15 Cifar SF  Byzantine Attacks {iidString} {experimentString}"),
            #([1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 20, 23, 26, 29], [], [], f"16  epsilon5.5 Cifar ALIE Byzantine Attacks{iidString} {experimentString}"),
             #([1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 23, 26, 29], [], [], f"17 Byzantine Attacks {iidString} {experimentString}"),
           #([1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 26, 29], [], [], f"18 Cifar SF Byzantine Attacks {iidString} {experimentString}"),
            # ([1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 25,26, 29], [], [], f"faulty_19 {iidString} {experimentString}"),
             #([1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 25,26, 28,29], [], [], f"20 Byzantine Attacks {iidString} {experimentString}"),
             #([1,2,3, 4,5,6, 7,8, 10,11,12, 13,14, 16,17,18, 19,20,21, 22,23,24, 25,26,27, 28,29,30], [], [], f"30 Byzantine Attacks {iidString} {experimentString}"),
            
            
        #([], [], [2], f"1 A Little Is Enough Attack {iidString} {experimentString}"),
             
        #([], [], [2, 5, 8], f"3 A Little Is Enough Attack {iidString}{experimentString} "),
            
        #([], [], [2, 5, 8, 11, 14, 17], f"6 A Little Is Enough Attack {iidString}{experimentString}"),
           
        #([], [], [5,8, 11,14, 17, 20, 23, 26,29], f"9 A Little Is Enough Attack {iidString} {experimentString}"),
            
        #([], [], [1,2, 4,5, 8, 11, 14, 17, 20, 23, 26, 29], f"12 A Little Is Enough Attack {iidString} {experimentString}"),
            
        #([], [], [1,2, 4,5, 7,8, 10,11, 13,14, 17,19, 20, 23, 26,], f"15 A Little Is Enough Attack {iidString} {experimentString}"),
             
        #([], [], [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 26, 29], f"18 A Little Is Enough Attack{iidString} {experimentString}"),
            
        #([], [], [2], f"1 Inner Product Manipulation Attack {iidString} {experimentString}"),
             
        #([], [], [2, 5, 8], f"3 Inner Product Manipulation Attack {iidString}{experimentString} "),
            
        #([], [], [2, 5, 8, 11, 14, 17], f"6 Inner Product Manipulation Attack {iidString}{experimentString}"),
           
        #([], [], [5,8, 11,14, 17, 20, 23, 26,29], f"9 Inner Product Manipulation Attack {iidString} {experimentString}"),
            
        #([], [], [1,2, 4,5, 8, 11, 14, 17, 20, 23, 26, 29], f"12 Inner Product Manipulation Attack {iidString} {experimentString}"),
            
        #([], [], [1,2, 4,5, 7,8, 10,11, 13,14, 17,19, 20, 23, 26,], f"15 Inner Product Manipulation Attack{iidString} {experimentString}"),
             
        #([], [], [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 26, 29], f"18 Inner Product Manipulation Attack{iidString} {experimentString}"),
           
            
            
            
            
             #([], [2, ], [], f"1 DBA Synthi Attack  {iidString} {experimentString}"),
             #([], [2, 5], [], f"2 DBA SYnth {iidString} {experimentString}"),
      #([], [2,5,8], [], f"3 F3BA CIFAR10 {iidString} {experimentString} "),
          #([], [2, 4, 5, 6,], [], f" 4 noSynth ALIE Cifar10 Synthi {iidString} {experimentString}"),
             #([], [2, 5, 8, 11, 14], [], f"5 Label Flipping Attacks {iidString} {experimentString}"),
    #([], [2,4, 5, 8, 11, 14], [], f"6 F3BA CIFAR10 {iidString} {experimentString}"),
            #([], [2, 5, 8, 11, 14, 17, 20], [], f"7 Label Flipping Attacks  {iidString} {experimentString}"),
         #([], [2, 3, 5, 8, 10, 11, 14, 23], [], f"8 Emnist Trojan{iidString} {experimentString}"),

     ([], [2,5, 6,8, 11, 12,14,], [], f"9 F3BA CIFAR10{iidString} {experimentString}"),# CIFAR10 [2,4,5, 6, 8, 10, 11, 12,14, ], Emnist[2, 5, 8, 11, 14, 17, 20, 23, 26,]
           #( [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29],    [],     f"10 Label Flipping Attacks {iidString} " ),
             #([], [2, 5, 8, 11, 14, 17,16, 20, 23, 26, 29], [], f"Emnist mal_11 {iidString} {experimentString}"),

      #([], [2,5, 8,10, 11, 14,16, 17, 20, 23, 26, 29], [], f"12 F3BA Cifar10 {iidString} {experimentString}"),
             #([], [2, 4,5, 7,8, 11, 14,16, 17, 20, 23, 26, 29], [], f"mal_13 {iidString} {experimentString}"),
            # ([], [2, 4,5, 7,8, 10,11, 14,16, 17, 20, 23, 26, 29], [], f"mal_14 SF {iidString} {experimentString}"),
      #([], [2,4,5, 7,8,10, 11, 14,16, 17, 19,20, 23, 26, 29] , [], f" 15 SF eMnist{iidString} {experimentString}"),
       #([], [2, 4,5, 7,8, 10,11, 13,14, 16,17, 20, 23, 26, 29], [], f"16 F3DB Cifar{iidString} {experimentString}"),
       # ([], [2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20,22, 23, 26, 29], [], f"17 F3DB cifar without WAMC{iidString} {experimentString}"),
   #([], [ 2,4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23,25, 26,29], [], f"18 SF EMNIST {iidString} {experimentString}"),
             #([], [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 25,26, 29], [], f"mal_19 {iidString} {experimentString}"),
             #([], [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 25,26, 28,29], [], f"mal_20 triger1{iidString} {experimentString}"),
            # ([2, ], [17, ], [], f"dual_faulty1_mal1 {iidString} {experimentString}"),
            # ([2, 5, ], [17, 20, ], [], f"dual_faulty2_mal2 {iidString} {experimentString}"),
            # ([2, 5, 8, ], [17, 20, 23, ], [], f"dual_faulty3_mal3 {iidString} {experimentString}"),
             #([2, 5, 8, 11, ], [17, 20, 23, 26], [], f"dual_faulty4_mal4 {iidString} {experimentString}"),
         #( [ 2, 5, 8,11,14,], [17, 20, 23, 26, 29],[],  f"5 Byzantine Attacks & 5 Label Flipping Attacks {iidString} ",  ),
            # ([1,2, 5, 8, 11, 14, ], [16,17, 20, 23, 26, 29], [], f"dual_faulty6_mal6 {iidString} {experimentString}"),
             #([1,2, 4,5, 8, 11, 14, ], [16,17, 19,20, 23, 26, 29], [], f"dual_faulty7_mal7 {iidString} {experimentString}"),
            # ([1,2, 4,5, 7,8, 11, 14, ], [16,17, 19,20, 22,23, 26, 29], [], f"dual_faulty8_mal8 {iidString} {experimentString}"),
             #([10,2, ], [ 4,5, 25,26, 29], [], f"dual_faulty2_mal3 {iidString} {experimentString}"),
            #([1,2, 4,5, 7,8, 10,11, 13,14, ], [16,17, 19,20, 22,23, 25,26, 28,29], [], f"dual_faulty10_mal10 {iidString} {experimentString}"),
        ]

        self.percUsers = torch.tensor(PERC_USERS, device=self.aggregatorConfig.device)
        # FedAvg, COMED, MKRUM, FedMGDA+, AFA
        # self.aggregators = [FAAggregator, FedDFAggregator, FedDFmedAggregator, FedBEAggregator, FedBEmedAggregator]
        self.aggregators = [

            #RUFLAggregator,
            #RKDAggregator,
            #RKDwithoutHDBSCANAggregator,
            #RA_without_HDBSCAN_KDAggregator,
            #RKDwithoutAWMCAggregator,
            #RAwithout_AWMC_KDAggregator,
            #TMAggregator,
            #SMAggregator,
            #RKDwithoutPMAggregator,

            
            #RKDwithoutAWMCAggregator,
            #RKDwithoutHDBSCANAWMCAggregator,
            #without_KDAggregator,
            
            #RFCLAggregator,
            
            FedAvgAggregator,
            FLAMEAggregator,
            FedDFAggregator,
            FedRADAggregator,
            FedBEAggregator,
            RLRAggregator,
            FoolsgoldAggregator,
            


            #MedianAggregator,
            #MKrumAggregator,
            #AFAAggregator,
            #FedMGDAplusAggregator,
            #CenteredClippingAggregator,
           
            #ModAFAAggregator,
            #RFCL_Without_PCAAggregator,
            #RFCL_With_FedAvg_Internal_AggAggregator,
  
            #FedRADAggregator,
            #FedDFAggregator,
            #FedDFmedAggregator,

            #KMeansAggregator,
            #HDBSCANAggregator,
            #AgglomerativeAggregator,
                     
        ]

    def scenario_conversion(self):
        """
        Sets the faulty, malicious and free-riding clients appropriately.

        Sets the config's and aggregatorConfig's names to be the attackName.
        """
        for faulty, malicious, freeRider, attackName in self.scenarios:

            self.faulty = faulty
            self.malicious = malicious
            self.freeRiding = freeRider
            self.name = attackName
            self.aggregatorConfig.attackName = attackName

            yield attackName


# Determines how much data each client gets (normalised)
PERC_USERS: List[float] = [
    0.2,
    0.15,
    0.2,
    0.2,
    0.1,
    0.15,
    0.1,
    0.15,
    0.2,
    0.2,
    0.2,
    0.3,
    0.2,
    0.2,
    0.1,
    #0.1,
    #0.1,
    #0.15,
    #0.2,
    #0.2,
    #0.1,
    #0.15,
    #0.2,
    #0.2,
    #0.1,
    #0.15,
    #0.1,
    #0.15,
    #0.2,
    #0.2,
]
