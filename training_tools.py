import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class LMCLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, margin=0.35, scale=30.0, W = None):
        super().__init__()
        self.margin = margin
        self.scale = scale
        # Das ist deine Shared Weight Matrix W (normalisiert)
        if W is not None:
            self.W = nn.Parameter(W)
            print("W was initialized by zero-shot alignment")
        else:
            self.W = nn.Parameter(torch.randn(embedding_dim, num_classes))
            print("W was initialized randomly")
        
    def forward(self, features, labels, subroute_class_indices=None):
        # Features normalisieren (L2)
        features = F.normalize(features, p=2, dim=1)

        if subroute_class_indices is not None:
            # --- HIER IST DER KNIFF ---
            # Wir schneiden uns NUR die Spalten aus W heraus, die zur Subroute gehören
            print("Der korrekte forward pass wurde gewaehlt")
            W_sub = self.W[:, subroute_class_indices]
            print(np.shape(W_sub))
            W_norm = F.normalize(W_sub, p=2, dim=0)
            
            # Die Logits haben jetzt nur noch die Dimension [Batch_Size, Klassen_in_Subroute] (z.B. 16 x 198)
            logits = torch.matmul(features, W_norm)
            
            # Da die Logits geschrumpft sind, stimmen die globalen Labels nicht mehr.
            # Wir müssen die globalen Labels (z.B. 0, 10, 20) auf lokale Indizes (0, 1, 2) mappen.
            mapping = {global_id: local_id for local_id, global_id in enumerate(subroute_class_indices)}
            local_labels = torch.tensor([mapping[label.item()] for label in labels], device=features.device)
            
            # Margin abziehen (auf den lokalen Indizes)
            for i in range(len(local_labels)):
                logits[i, local_labels[i]] -= self.margin
                
            return F.cross_entropy(logits * self.scale, local_labels)
        
        else:
            W_norm = F.normalize(self.W, p=2, dim=0)
            
            # Kosinus-Ähnlichkeit berechnen
            logits = torch.matmul(features, W_norm)
            
            # Margin vom Target-Logit abziehen (NocPlace Logik)
            for i in range(len(labels)):
                logits[i, labels[i]] -= self.margin
                
            # Skalieren für die Softmax
            return F.cross_entropy(logits * self.scale, labels)
    
class IKTLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temp = temperature
        self.kl = nn.KLDivLoss(reduction="batchmean")
        
    def forward(self, night_logits, day_logits):
        # Softmax mit Temperatur auf den Lehrer (Day) anwenden
        p_day = F.softmax(day_logits / self.temp, dim=1)
        # Log-Softmax auf den Schüler (Night) anwenden
        log_p_night = F.log_softmax(night_logits / self.temp, dim=1)
        
        # KL-Divergenz berechnen und skalieren
        return self.kl(log_p_night, p_day) * (self.temp ** 2)
    




