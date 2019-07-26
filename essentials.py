import numpy as np
import matplotlib.pyplot as plt
import os, random
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time
import matplotlib.image as mp
import argparse
def model(arch, hiddenlayer_1, hiddenlayer_2):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        inputs = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        inputs = 1024
    else:
        print('Unknown arch')
    classifier = nn.Sequential(nn.Linear(inputs, hiddenlayer_1),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hiddenlayer_1, hiddenlayer_2),
        nn.LogSoftmax(dim=1))
    model.classifier = classifier   
    return model
def model_trainer(model,epochs,train_loader,valid_loader,criterion,optimizer,device):
    if device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    print_every = 5
    steps = 0
    accuracy = 0
    print('Starting Training UwU')
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            a=time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0
            
            
                for ii, (inputs2,labels2) in enumerate(valid_loader):
                    optimizer.zero_grad()
                
                    inputs2, labels2 = inputs2.to(device) , labels2.to(device)
                    model.to(device)
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                vlost = vlost / len(valid_loader)
                accuracy = accuracy /len(valid_loader)
                b=time.time()-a
                b= b/60
                    
             
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Lost {:.4f}".format(vlost),
                   "Accuracy: {:.4f}%".format(accuracy*100),
                     "Time: {:.4f}".format(b))
                                
            
                running_loss = 0
                model.train()
    
    print('Done Training !w!')
    return model
def data_loader(arg):
    data_dir = arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Apply the required transfomations to the test dataset in order to maximize the efficiency of the learning
    #process


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Crop and Resize the data and validation images in order to be able to be fed into the network

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # The data loaders are going to use to load the data to the NN(no shit Sherlock)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)



    return trainloader , valid_loader, testloader
def save(model,arg):
    checkpoint = {'arch' : arg.arch, 
              'state_dict':model.state_dict(),}
    torch.save(checkpoint, 'classifier.pth')
    print(model.state_dict)
    print("Done Saving !w!")
    return checkpoint
def model_loader(filepath, hiddenlayer_1,hiddenlayer_2,arch):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        inputs = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        inputs = 1024
    else:
        print('Unknown arch')
    model = models.vgg16(pretrained=True)
    checkpoint = torch.load(filepath)
    classifier = nn.Sequential(nn.Linear(inputs, hiddenlayer_1),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hiddenlayer_1, hiddenlayer_2),
        nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    print('Done Loading in !w!')
    return model
def model_tester(test_loader,model,device):
    if str(device) == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.eval()
    accuracy = 0
    
    model.to(device)
    pass_count = 0
    done = 0
    for ii, (inputs,labels) in enumerate(test_loader):
        Start_time= time.time()
        pass_count += 1
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        Overall_time=time.time()-Start_time
        if pass_count == 5:
            print("Testing Accuracy: {:.4f}".format(accuracy/pass_count),
            ("Time: {:.4f}").format(Overall_time/60))
            pass_count=0
            done += 1
            accuracy = 0
            if done == 5:
                print('Done Testing OwO')
                break
def predict(image_path, model, topk, device):
    if str(device) == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    with torch.no_grad():    
        if str(device) == 'cuda':
            output = model.forward(img_torch.cuda())
        else:
            output=model.forward(img_torch)
    
    probability = F.softmax(output.data,dim=1)
    return probability.topk(topk)
def process_image(image_path):
    '''
    Arguments: The image's path
    Returns: The image as a tensor
    This function opens the image usign the PIL package, applies the  necessery transformations and returns the image as a tensor ready to be fed to the network
    '''

    path = str(image_path)
    img = Image.open(image_path) # Here we open the image

    make_img_good = transforms.Compose([ # Here as we did with the traini ng data we will define a set of
        # transfomations that we will apply to the PIL image
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_image = make_img_good(img)

    return tensor_image


    