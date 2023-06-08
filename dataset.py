import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

'''Split dataset into D_shadow and D_target'''
def split_D_shadow(train_dataset,batch_size=256):
    size=len(train_dataset)
    indice=list(range(size))
    d1=int(np.floor(size/4))
    D_train_shadow=indice[:d1]
    D_out_shadow=indice[d1:d1*2]
    D_train_shadow_loader=DataLoader(train_dataset, batch_size, sampler=SubsetRandomSampler(D_train_shadow))
    D_out_shadow_loader=DataLoader(train_dataset, batch_size, sampler=SubsetRandomSampler(D_out_shadow))
    return D_train_shadow_loader, D_out_shadow_loader

def split_D_target(train_dataset,batch_size=256):
    size=len(train_dataset)
    indice=list(range(size))
    d1=int(np.floor(size/4))
    target_0=indice[2*d1:3*d1]
    target_1=indice[3*d1:]
    D_target_in=DataLoader(train_dataset,batch_size, sampler= SubsetRandomSampler(target_0))
    D_target_out=DataLoader(train_dataset,batch_size, sampler=SubsetRandomSampler(target_1))
    return D_target_in, D_target_out

'''Obtain the top3 possible attributes '''
def feature_vector(shadow_model, batch, N):
    batch = batch.cpu()
    shadow_model = shadow_model.cpu()
    output = shadow_model(batch)
    probability = F.softmax(output, dim=-1)
    sorted_probs, indices = torch.sort(probability, descending=True)
    top3 = torch.narrow(sorted_probs, dim=1, start=0, length=3)
    return top3

