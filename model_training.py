
from functools import partial
from pyexpat import _Model
import numpy as np
import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from model import *
from utils import *


parser = arg_parse.get_args()
args = parser.parse_args()
def train_model(config,checkpoint_dir= None, data_dir = None):

    model = Net(config["l1"], config["l2"])
    model = resource_allocation(model)

    loss_fn = nn.CrossEntropyLoss()  # make it dynamic later
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9) #make it dynamic later

    trainset,testset = load_data(data_dir = '../data')
    dataset_sizes  = {x: len(x) for x in ['trainset', 'testset']}

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(args.epochs):

        training_loss = 0
        

        for i,data in enumerate(trainloader,0):

            inputs,labels = data
            inputs, labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs)
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            
            





        










