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
p.add_argument('--gpu', action='store_true', default='gpu')
p.add_argument('--epochs', type=int, default=2)
p.add_argument('--arch', type=str,default='vgg16')
p.add_argument('--learning_rate', type=float, default=0.001)
p.add_argument('--checkpoint', type=str)
p.add_argument('--criterion', type=float, default=nn.NLLLoss())
p.add_argument('--save_file', type=str, default='classifier.pth')
p.add_argument('--topk', type=float, default=5)
p.add_argument('--img_path', type=str,default='ImageClassifier/flowers/test/10/image_07090.jpg')
p.add_argument('--json', type=str, default='ImageClassifier/cat_to_name.json')
p.add_argument('--hiddenlayer1',type=int,default=4096)
p.add_argument('--hiddenlayer2', type=int,default=102)
arg = p.parse_args()
train_loader,test_loader,valid_loader = essentials.data_loader(arg)


model = essentials.model_loader(arg.save_file,arg.hiddenlayer1,arg.hiddenlayer2,arg.arch)


with open(arg.json, 'r') as json_file:
    cat_to_name = json.load(json_file)


probabilities = essentials.predict(arg.img_path, model, arg.topk, arg.gpu)


labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])


i = 0
while i < arg.topk:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

print("Done!!!!, OwO")


