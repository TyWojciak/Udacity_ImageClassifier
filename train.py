import numpy as np
import matplotlib.pyplot as plt
import os, random
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time
import matplotlib.image as mp
import argparse
import essentials
p = argparse.ArgumentParser()
p.add_argument('--data_dir', type=str, default='ImageClassifier/flowers')
p.add_argument('--gpu', action='store_true', default='cuda')
p.add_argument('--epochs', type=int, default=2)
p.add_argument('--arch', type=str,default='vgg16')
p.add_argument('--learning_rate', type=float, default=0.001)
p.add_argument('--checkpoint', type=str)
p.add_argument('--criterion', type=float, default=nn.NLLLoss())
p.add_argument('--save_file', type=str, default='classifier.pth')
p.add_argument('--hiddenlayer1',type=int,default=4096)
p.add_argument('--hiddenlayer2', type=int,default=102)
arg = p.parse_args()

train_loader,test_loader,valid_loader = essentials.data_loader(arg)
model = essentials.model(arg.arch,arg.hiddenlayer1,arg.hiddenlayer2)
model = essentials.model_loader(arg.save_file,arg.hiddenlayer1,arg.hiddenlayer2,arg.arch)
optimizer = optim.Adam(model.classifier.parameters(), lr=arg.learning_rate)
model = essentials.model_trainer(model,arg.epochs,train_loader,valid_loader,arg.criterion,optimizer, arg.gpu)
essentials.model_tester(train_loader,model, arg.gpu)
checkpoint = essentials.save(model,arg)
    
