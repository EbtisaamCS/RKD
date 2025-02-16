import os

from torch import optim
from utils.typings import BlockedLocations, Errors, Accuracy, FreeRiderAttack, PersonalisationMethod, AttackSuccesRate, attack_success_variance 

from datasetLoaders.DatasetInterface import DatasetInterface
from experiment.CustomConfig import CustomConfig
from typing import Callable, Dict, List, NewType, Optional, Tuple, Dict, Type
import json
from loguru import logger

from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
from datasetLoaders.MNIST import DatasetLoaderMNIST
from datasetLoaders.CIFAR10 import DatasetLoaderCIFAR10
from datasetLoaders.FashionMNIST import DatasetLoaderFashionMNIST
from datasetLoaders.EMNIST import DatasetLoaderEMNIST
#from datasetLoaders.TinyImageNet import DatasetLoaderTinyImageNet
from aggregators. ModAFA import ModAFAAggregator
from aggregators.RUFL import RUFLAggregator


from aggregators.FedRAD import FedRADAggregator
from aggregators.FedBE import FedBEAggregator
from aggregators.FedDF import FedDFAggregator

#from datasetLoaders.COVID19 import DatasetLoaderCOVID19

from classifiers import MNIST
from classifiers import CIFAR10
#from classifiers import FashionMNIST
from classifiers import EMNIST
#from classifiers import TinyImageNet


#from classifiers import Pneumonia
#from classifiers import CovidNet

from logger import logPrint
from client import Client

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import time
import gc
from torch import cuda, Tensor, nn

from aggregators.Aggregator import Aggregator, allAggregators
#from aggregators.AFA import AFAAggregator
#from aggregators.FedMGDAplus import FedMGDAplusAggregator
#from aggregators.RFA import RFAAggregator
from aggregators.MKrum import MKrumAggregator
#from aggregators.CC import CenteredClippingAggregator
from aggregators.Foolsgold import FoolsgoldAggregator
from aggregators.FLAME import FLAMEAggregator
from aggregators.RLR import RLRAggregator
from aggregators.FedAvg import FedAvgAggregator
##from aggregators.FedRAD import FedRADAggregator 

from aggregators.AFA import AFAAggregator
from aggregators.FedMGDAplus import FedMGDAplusAggregator
from aggregators.Median import MedianAggregator
from aggregators.MKrum import MKrumAggregator
from aggregators.CC import CenteredClippingAggregator


#from aggregators.RKD import RKDAggregator
from aggregators.RKDwithoutPM import RKDwithoutPMAggregator
from aggregators. RA_without_HDBSCAN_KD import RA_without_HDBSCAN_KDAggregator
from aggregators. RAwithout_AWMC_KD import RAwithout_AWMC_KDAggregator

from aggregators.RKDwithoutHDBSCAN import RKDwithoutHDBSCANAggregator
from aggregators. RKDwithoutHDBSCANAWMC import RKDwithoutHDBSCANAWMCAggregator
from aggregators.RKDwithoutAWMC import RKDwithoutAWMCAggregator
from aggregators.TM import TMAggregator
from aggregators.SM import SMAggregator

from aggregators.without_KD import without_KDAggregator



from aggregators.RFCL import RFCLAggregator
#from aggregators.RFCL_With_FedAvg_Internal_Aggregator import RFCL_With_FedAvg_Internal_AggAggregator
#from aggregators.RFCL_Without_PCA import RFCL_Without_PCAAggregator
#from aggregators.KMeans import KMeansAggregator
#from aggregators.HDBSCAN import HDBSCANAggregator
#from aggregators.Agglomerative import AgglomerativeAggregator
#from aggregators.FedDF import FedDFAggregator
#from aggregators.FedDFmed import FedDFmedAggregator





from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('large')




SEED = 0

# Colours used for graphing, add more if necessary
COLOURS: List[str] = [

    "tab:purple",
    "tab:green",
    "tab:cyan",
    
    "tab:orange",
    "tab:brown",
    "tab:blue",
    "tab:pink",
    
    
    "gold",
    "chartreuse",
    "saddlebrown",
    "blueviolet",
    "navy",
    "cornflowerblue",
    "thistle",
    "dodgerblue",
    "crimson",
    "darkseagreen",
    "maroon",
    "mediumspringgreen",
    "burlywood",
    "olivedrab",
    "linen",
    "mediumorchid",
    "teal",
    "black",
    "gold",
]
#linestyles : List[str]  = ['-', '-', '-', '-','-','-.','--']
#markers : List[str] = ['D', '*', '+', '','o','','']

linestyles = ['-.', '--', '-', ':', '--','-.',':','-.', '--', '-', ':', '--','-.',':']
markers = ['D', 's', '^', 'd', 'o','*','+','s', '^', 'd', 'o','*','+']
   

def __experimentOnMNIST(
    config: DefaultExperimentConfiguration, title="", filename="", folder="DEFAULT"
) ->Tuple[Dict[str, Errors], Dict[str, AttackSuccesRate]]: 
    """
    MNIST Experiment with default settings
    """
    #dataLoader = DatasetLoaderMNIST().getDatasets
    #classifier = MNIST.Classifier
    
    dataLoader = DatasetLoaderCIFAR10().getDatasets
    classifier = CIFAR10.Classifier
    
    #dataLoader = DatasetLoaderTinyImageNet().getDatasets
    #classifier = TinyImageNet.Classifier

    #dataLoader = DatasetLoaderEMNIST().getDatasets
    #classifier = EMNIST.Classifier
   
    #dataLoader = DatasetLoaderFashionMNIST().getDatasets
    #classifier = FashionMNIST.Classifier
   
    
    return __experimentSetup(config, dataLoader, classifier, title, filename, folder)




def __experimentSetup(
    config: DefaultExperimentConfiguration,
    
    datasetLoader: Callable[
        [Tensor, Tensor, Optional[Tuple[int, int]]], Tuple[List[DatasetInterface], DatasetInterface]
    ],
    classifier,
    title: str = "DEFAULT_TITLE",
    filename: str = "DEFAULT_NAME",
    folder: str = "DEFAULT_FOLDER",
) -> Tuple[Dict[str, Errors], Dict[str, AttackSuccesRate]]: 
    __setRandomSeeds()
    gc.collect()
    cuda.empty_cache()
    errorsDict: Dict[str, Errors] = {}
    AttackSuccesRateDict: Dict[str, AttackSuccesRate] = {}

    

    for aggregator in config.aggregators:
        
        name = aggregator.__name__.replace("Aggregator", "")
        name = name.replace("Plus", "+")
        logPrint("TRAINING {}".format(name))
        
        if config.privacyPreserve is not None:
            #errors, Accuracy, AttackSuccesRate 
            errors,AttackSuccesRate = __runExperiment(
                config, datasetLoader, classifier, aggregator, config.privacyPreserve, folder
            )
        else:
            #errors, Accuracy, AttackSuccesRate
            errors,AttackSuccesRate = __runExperiment(
                config, datasetLoader, classifier, aggregator, False, folder
            )
            logPrint("TRAINING {} with DP".format(name))
            errors,AttackSuccesRate = __runExperiment(
                config, datasetLoader, classifier, aggregator, True, folder
            )



        
        #errorsDict[name] = errors.tolist()
        errorsDict[name] = errors.tolist()
        #AccuracyDict[name] = Accuracy.tolist()
        AttackSuccesRateDict[name] = AttackSuccesRate.tolist()
        #attack_success_varianceDict[name] = attack_success_variance.tolist()

     

    # Writing the attack_success_rate and errors to json files for later inspection.
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if not os.path.isdir(f"{folder}/json"):
        os.mkdir(f"{folder}/json")
    if not os.path.isdir(f"{folder}/graphs"):
        os.mkdir(f"{folder}/graphs")
    with open(f"{folder}/json/{filename} attack_success_rate(Seed: {SEED}).json", "w+") as outfile:
        json.dump(AttackSuccesRateDict, outfile)
    with open(f"{folder}/json/{filename} errors (Seed: {SEED}).json", "w+") as outfile:
        json.dump(errorsDict, outfile)
    
   

     
    # Plots the individual aggregator errors and attack success rates
    if config.plotResults:
    # Plot for Error Rate
        plt.figure()
        i = 0
        for name, err in errorsDict.items():
            plt.plot(err, color=COLOURS[i], linestyle=linestyles[i])
            i += 1
        plt.legend(errorsDict.keys(),prop = fontP)
        plt.xlabel(f"Rounds")
        plt.ylabel("Accuracy Rate (%)")
        plt.title(title, loc="center", wrap=True)
        #plt.ylim(0.5, 1.0)
        #plt.ylim([0.0, 1.0])
        offset = 0.05  # or whatever small value you find appropriate
        plt.ylim([-offset, 1.0 + offset])
        plt.savefig(f"{folder}/graphs/{filename}_error_rate.pdf",bbox_inches="tight", pad_inches=4.3)
        


    # Plot for Attack Success Rate
        plt.figure()
        i = 0
        for name, asr in AttackSuccesRateDict.items():
            plt.plot(asr, color=COLOURS[i],linestyle=linestyles[i])
            i += 1
        plt.legend(AttackSuccesRateDict.keys(),prop = fontP)
        plt.xlabel(f"Rounds")
        plt.ylabel("Attack Success Rate (%)")
        plt.title(title, loc="center", wrap=True)
        #plt.ylim([0.0, 1.0])
        offset = 0.05  # or whatever small value you find appropriate
        plt.ylim([-offset, 1.0 + offset])
        plt.savefig(f"{folder}/graphs/{filename}_attack_success_rate.pdf",bbox_inches="tight", pad_inches=4.3)

    # Plot for attack_success_variance
        #plt.figure()
        #I = 0
        #for name, acr in attack_success_varianceDict.items():
            #plt.plot(acr, color=COLOURS[i])
            #I += 1
        #plt.legend(attack_success_varianceDict.keys(),prop = fontP)
        #plt.xlabel(f"Rounds - {config.epochs} Epochs per Round")
        #plt.ylabel("Attack Success Variance (%)")
        #plt.title(title, loc="center", wrap=True)
        #offset = 0.05  # or whatever small value you find appropriate
        #plt.ylim([-offset, 1.0 + offset])
        #plt.savefig(f"{folder}/graphs/{filename} attack_success_variance.pdf",bbox_inches="tight", pad_inches=4.3)

    return errorsDict, AttackSuccesRateDict



def __runExperiment(
    config: DefaultExperimentConfiguration,
    datasetLoader: Callable[
        [Tensor, Tensor, Optional[Tuple[int, int]]], Tuple[List[DatasetInterface], DatasetInterface]
    ],
    classifier: nn.Module,
    agg: Type[Aggregator],
    useDifferentialPrivacy: bool,
    folder: str = "test",
) -> Tuple[Errors, AttackSuccesRate]: #, #AttackSuccesRate]:
    """
    Sets up the experiment to be run.

    Initialises each aggregator appropriately
    """
    serverDataSize = config.serverData
    if not agg.requiresData():
        print("Type of agg:", type(agg))
        print("agg:", agg)
        serverDataSize = 0
        
    trainDatasets, testDataset, serverDataset= datasetLoader(
        config.percUsers,
        config.labels,
        config.datasetSize,
        config.nonIID,
        config.alphaDirichlet,
        serverDataSize,
    )
    # TODO: Print client data partition, i.e. how many of each class they have. Plot it and put it in report.

   

    #clientPartitions = torch.stack([torch.bincount(t.dataset.get_labels(), minlength=10) for t in trainDatasets])

    clientPartitions = torch.stack([torch.bincount(t.labels, minlength=10) for t in trainDatasets])#Emnist and Cifar10
    #clientPartitions = torch.stack([torch.bincount(t.get_labels(), minlength=10) for t in trainDatasets]) #Cifar10

    
    logPrint(
        f"Client data partition (alpha={config.alphaDirichlet}, percentage on server: {100*serverDataSize:.2f}%)"
   )
    logPrint(f"Data per client: {clientPartitions.sum(dim=1)}")
    logPrint(f"Number of samples per class for each client: \n{clientPartitions}")
    plt.rcParams.update({'font.size': 18})
    
    #plt.figure(figsize=(9,10))
   
    #plt.hist([torch.bincount(t.labels, minlength=10) for t in trainDatasets], stacked=True, 
              #bins = 2,
            #bins=np.arange(31),
            #label=["{}".format(i) for i in range(9)], rwidth=0.5)
    #plt.xticks(np.arange(30))
    #plt.xticks (range(30))
    #plt.xlabel(f"Clients")
    #plt.ylabel(f"Dataset Size")
    #plt.legend()
    #plt.savefig(f'{config.alphaDirichlet}.pdf',bbox_inches="tight",pad_inches=0.1)
    #plt.savefig(f'{config.alphaDirichlet}.pdf')
   
   
    plt.show()
    
    clients = __initClients(config, trainDatasets, useDifferentialPrivacy)
    # Requires model input size update due to dataset generalisation and categorisation
    if config.requireDatasetAnonymization:
        classifier.inputSize = testDataset.getInputSize()
    model = classifier().to(config.aggregatorConfig.device)
    name = agg.__name__.replace("Aggregator", "")
    
    aggregator = agg(clients, model, config.aggregatorConfig)

  
    if isinstance(aggregator, AFAAggregator):
        aggregator.xi = config.aggregatorConfig.xi
        aggregator.deltaXi = config.aggregatorConfig.deltaXi
    
    #elif isinstance(aggregator, RFCL_With_FedAvg_Internal_AggAggregator):
        #aggregator._init_aggregatorsfed(config.internalfedAggregator, config.externalfedAggregator)
       
        
    #elif isinstance(aggregator, RFCL_Without_PCAAggregator):
        #aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)
    
    
    elif isinstance(aggregator, RFCLAggregator):
        aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)
        
        
    #elif isinstance(aggregator, KMeansAggregator):
        #aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)
        
    #elif isinstance(aggregator, HDBSCANAggregator):
        #aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)
   
    #elif isinstance(aggregator, AgglomerativeAggregator):
         #aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)
   
  
  
    if aggregator.requiresData():
        serverDataset.data = serverDataset.data.to(aggregator.config.device)
        serverDataset.labels = serverDataset.labels.to(aggregator.config.device)
        aggregator.distillationData = serverDataset


    errors, AttackSuccesRate  = aggregator.trainAndTest(testDataset)
    #errors , Accuracy , AttackSuccesRate = aggregator.trainAndTest(testDataset, poisoned_Datasets)
    
    return errors ,  AttackSuccesRate

     


def __initClients(
    config: DefaultExperimentConfiguration,
    trainDatasets: List[DatasetInterface],
    useDifferentialPrivacy: bool,
) -> List[Client]:
    """
    Initialises each client with their datasets, weights and whether they are not benign
    """
    usersNo = config.percUsers.size(0)
    p0 = 1 / usersNo
    logPrint("Creating clients...")
    clients: List[Client] = []
    logPrint("clients...", clients)
    for i in range(usersNo):
        clients.append(
            Client(
                idx=i,
                trainDataset=trainDatasets[i],
                epochs=config.epochs,
                batchSize=config.batchSize,
                learningRate=config.learningRate,
                p=p0,
                alpha=config.alpha,
                beta=config.beta,
                Loss=config.Loss,
                Optimizer=config.Optimizer,
                device=config.aggregatorConfig.device,
                useDifferentialPrivacy=useDifferentialPrivacy,
                epsilon1=config.epsilon1,
                epsilon3=config.epsilon3,
                needClip=config.needClip,
                clipValue=config.clipValue,
                needNormalization=config.needNormalization,
                releaseProportion=config.releaseProportion,
            )
        )

    nTrain = sum([client.n for client in clients])
    # Weight the value of the update of each user according to the number of training data points
    for client in clients:
        client.p = client.n / nTrain

    # Create malicious (byzantine) and faulty users
    for client in clients:
        if client.id in config.faulty:
            client.byz = True
            logPrint("User", client.id, "is faulty.")
        if client.id in config.malicious:
            client.flip = True
            logPrint("User", client.id, "is malicious.")
            #client.trainDataset.zeroLabels()
            #client.trainDataset.setLabels(3)
        logPrint("clients...", client.id)

           
    return clients


def __setRandomSeeds(seed=SEED) -> None:
    """
    Sets random seeds for all of the relevant modules.

    Ensures consistent and deterministic results from experiments.
    """
    print(f"Setting seeds to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed(seed)


def experiment(exp: Callable[[], None]):
    """
    Decorator for experiments so that time can be known and seeds can be set

    Logger catch is set for better error catching and printing but is not necessary
    """

    @logger.catch
    def decorator():
        __setRandomSeeds()
        logPrint("Experiment {} began.".format(exp.__name__))
        begin = time.time()
        exp()
        end = time.time()
        logPrint("Experiment {} took {}".format(exp.__name__, end - begin))

    return decorator


@experiment
def program() -> None:
    """
    Main program for running the experiments that you want run.
    """
    config = CustomConfig()


    for attackName in config.scenario_conversion():
        #errors
        errors , AttackSuccesRate = __experimentOnMNIST(
            config,
            title=f"",
            filename=f"{attackName}",
            folder=f"test",
        )


# Running the program here
program()


