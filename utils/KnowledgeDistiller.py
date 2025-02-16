
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from logger import logPrint
from typing import List
import gc
from torch.nn.utils import parameters_to_vector
from torch import nn, optim, Tensor

from torch.nn import Module
from torch.nn.utils import parameters_to_vector





class KnowledgeDistiller:
    """
    A class for Knowledge Distillation using ensembles.
    """

    def __init__(
        self,
        dataset,
        epoc=3,
        batch_size=16,
        temperature=2,  
        method="avglogits",
        device="cpu",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.T = temperature
        self.epoc = epoc
        self.lr1 = 0.001
        self.swa_lr = 0.01
      
        
        self.momentum = 0.9 
        
        self.method = method 
        self.device = device
        self.optimizer_type = "SGD" 
        


        
        
    
    def distillKnowledge(self, ensemble, Scoresmodel):
        """
        Takes in a teacher ensemble (list of models) and a student model.
        Trains the student model using unlabelled dataset, then returns it.
        Args:
            teacher_ensemble is list of models used to construct pseudolabels using self.method
            student_model is models that will be trained
        """

        gc.collect()
        torch.cuda.empty_cache()
        # Set labels as soft ensemble prediction
        self.dataset.labels = self._pseudolabelsFromEnsemble(ensemble, self.method)
        print("LR:----" ,self.lr1)
        print("LRSWA:----", self.swa_lr)
        print("Epoch:----",self.epoc)
        print("optimizer_type:----",self.optimizer_type)

        if self.optimizer_type == "SGD":
            opt = optim.SGD(Scoresmodel.parameters(), momentum=self.momentum, lr=self.lr1, weight_decay=1e-5)
        
        Loss = nn.KLDivLoss
        loss = Loss(reduction="batchmean")
     

        swa_model = AveragedModel(Scoresmodel)
        scheduler = CosineAnnealingLR(opt, T_max=100)
        swa_scheduler = SWALR(opt, swa_lr=self.swa_lr)

        dataLoader = DataLoader(self.dataset, batch_size=self.batch_size,)
        for i in range(self.epoc):
            
            total_err = 0
            for j, (x, y) in enumerate(dataLoader):

                
                opt.zero_grad()
                pred = Scoresmodel(x)


                err = loss(F.log_softmax(pred / self.T, dim=1), y) * self.T * self.T
                err.backward()
                total_err += err
                opt.step()
            logPrint(f"KD epoch {i}: {total_err}")
            scheduler.step()
            swa_model.update_parameters(Scoresmodel)
            swa_scheduler.step()

            torch.optim.swa_utils.update_bn(dataLoader, swa_model)


        gc.collect()
        torch.cuda.empty_cache()
        return swa_model.module
  

    
    
    
    
    def _pseudolabelsFromEnsemble(self, ensemble, method=None):
        """
        Combines the probabilities to make ensemble predictions.
        3 possibile methods:
            avglogits: Takes softmax of the average outputs of the models
            medlogits: Takes softmax of the median outputs of the models
            avgprob: Averages the softmax of the outputs of the models

        Idea: Use median instead of averages for the prediction probabilities!
            This might make the knowledge distillation more robust to confidently bad predictors.
        """
        if isinstance(ensemble, Module):
            # If it's a single model, convert it into a list
            ensemble = [ensemble]
        if method is None:
            method = self.method
        print (f" ensemble-------: {len(ensemble)}")
        print (f" Method-------: {len(method)}")
        

            
        with torch.no_grad():
            dataLoader = DataLoader(self.dataset, batch_size=self.batch_size)
            preds = []
            for i, (x, y) in enumerate(dataLoader):
       
                predsBatch = torch.stack([m(x) / self.T for m in ensemble])
                preds.append(predsBatch)
            preds = torch.cat(preds, dim=1)
            print(f"Final preds shape: {preds.shape}")


            
            
            if method == "avglogits":
                pseudolabels = preds.mean(dim=0)
                return F.softmax(pseudolabels, dim=1)
            elif method == "medlogits":
                pseudolabels, idx = preds.median(dim=0)
                return F.softmax(pseudolabels, dim=1)

            
            else:
                raise ValueError(
                    "pseudolabel method should be one of: avglogits, medlogits, avgprob"
                )
                

    


    
    
        
    def calculateModelReliability(self, ensemble: List[nn.Module], clients: List[nn.Module] = []) -> Tensor:
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            dataLoader = DataLoader(self.dataset, batch_size=self.batch_size)
            all_preds = []

            for i, (x, y) in enumerate(dataLoader):
                
                predsBatch = torch.stack([m(x) for m in ensemble])
                all_preds.append(predsBatch)

            all_preds = torch.cat(all_preds, dim=0)  # Shape: [num_models, num_data_points, num_classes]
            
            # Find the consensus predictions
            consensus_preds = torch.mode(all_preds.argmax(dim=2), dim=0).values

            # Calculate reliability scores based on agreement with the consensus
            reliability_scores = (all_preds.argmax(dim=2) == consensus_preds.unsqueeze(0)).float().sum(dim=1)
            

            
        return reliability_scores
    
    def weightedVotingBasedScores(self, ensemble: List[nn.Module], clients: List[nn.Module] = []) -> Tensor:
        logPrint(f"Calculating model scores based on weighted voting")

        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            dataLoader = DataLoader(self.dataset, batch_size=self.batch_size)
            preds = []
            
            # Get the scores from the new reliability calculation method
            reliability_scores = self.calculateModelReliability(ensemble, clients)

            for i, (x, y) in enumerate(dataLoader):
                
                predsBatch = torch.stack([m(x) / self.T * reliability_scores[i] for i, m in enumerate(ensemble)])
                preds.append(predsBatch)

            preds = torch.cat(preds, dim=1)
            
            # Here I'm assuming that you want to calculate 'idx' as the indices of max values from 'preds'.
            pseudolabels, idx = preds.sum(dim=0).max(dim=1) # Summing the weighted logits and getting the class with max sum # median

            
            client_p = torch.tensor([c.p for c in clients]).to(self.device)
            # Clamping the values in idx to be within the range of client_p size
            idx = idx.clamp(max=client_p.size(0)-1)

            counts = torch.bincount(idx.view(-1), minlength=client_p.size(0)).to(self.device)  #minlength=len(ensemble)
            print("counts values:", counts)
            



            counts_p = counts / counts.sum()

            logPrint("Counts:", counts)
            mask = torch.ones(counts.shape, dtype=bool)
            # mask[self.malClients] = True
            mask = mask.to(self.device)
            print(
                f"Mean of attackers: {counts_p[~mask].mean()*100:.2f}%, healthy: {counts_p[mask].mean()*100:.2f}%"
            )
            
            

            # If list of clients is provided, scale according to client.p
            if len(clients) > 0:
                #client_p = torch.tensor([c.p for c in clients]).to(self.device)
                print("Max index:", idx.max().item())

                print("Number of models in ensemble:", len(ensemble))
                print("Number of clients:", len(clients))
                print("Shape of counts_p:", counts_p.shape)
                print("Shape of client_p:", client_p.shape)
                counts_p *= client_p
                counts_p /= counts_p.sum()

            gc.collect()
            torch.cuda.empty_cache()
            return counts_p
