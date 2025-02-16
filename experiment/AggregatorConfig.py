import os

from utils.typings import FreeRiderAttack, PersonalisationMethod
from torch import device, cuda


class AggregatorConfig:
    """
    Configuration for the aggregators.

    Use this for information that you want the aggregator to know about.
    """

    def __init__(self):

        # Total number of training rounds
        self.rounds: int =30

        self.device = device("cuda" if cuda.is_available() else "cpu")
        #self.device = device("cpu")

        # Name of attack being employed
        self.attackName = ""

        # Pipeline config
        self.detectFreeRiders: bool = False
        self.freeRiderAttack: FreeRiderAttack = FreeRiderAttack.NOISY

        # Privacy Amplification settings  (Sets how many clients are sampled)
        self.privacyAmplification = False
        self.amplificationP = 0.3

        # FedMGDA+ Parameters:
        self.innerLR: float = 0.1

        #AFA Parameters:
        self.xi: float = 2
        self.deltaXi: float = 0.25
        
        #CC Parameters:
        self.agg_config = {}
        self.agg_config["clip_factor"] = 100.0


         #Clustering Config:
        self.cluster_count: int =3
        self.min_cluster_size=4
        self.hdbscan_min_samples=3
        self.cluster_distance_threshold=1.0
        
        
        self.personalisation: PersonalisationMethod = PersonalisationMethod.SELECTIVE
        self.threshold: bool = True

        # FedBE Parameters
        self.sampleSize = 15  # 15 Number of models sampled
        self.samplingMethod = "dirichlet"  # gaussian, dirichlet, dirichlet_elementwise
        self.samplingDirichletAlpha = 0.1  #
