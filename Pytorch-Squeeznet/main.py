#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import numpy as np
from torch.autograd import Variable
import model_squeeznet
import matplotlib.pyplot as plt


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

#The compose function allows for multiple transforms
#transforms.ToTensor() converts our PILImage to a tensor of shape (C x H x W) in the range [0,1]
#transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R, G, B)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))])

train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)

test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

from torch.utils.data.sampler import SubsetRandomSampler

#Training
n_training_samples = 50000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
print(train_sampler)

#Test
n_test_samples = 10000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

# Init Figure
avg_loss = list()
fig1, ax1 = plt.subplots()

#---------------------------TRAINING FUNCTION----------------------
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.view(len(target),-1)
        loss = F.nll_loss(output, target)

        # Conpute Average Loss
        avg_loss.append(loss.data[0])
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            ax1.plot(avg_loss)
            fig1.savefig("Net_loss.jpg")

#-----------------------------TEST FUNCTION-------------------------
def test(args, model, device, test_loader):
#    model.val()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.view(len(target),-1)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

def main():
    # # -----------------  Hyoerparameters  ---------------------
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 SqueezeNet')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=55, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    
    args = parser.parse_args()

    # Check having GPU or not
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # Split Train and test data
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=2)
    print(len(train_loader.dataset))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=2)

    # Init Model
    model = model_squeeznet.SqueezeNet().to(device)
    print(model)

    # Init Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # *************** Start Training model over each Epoch************
    since = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        torch.save(model, "akash_squeezenet")

    # Total Time for Training using GPU
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # ***************Start Test model From test Dataset***************
    test(args, model, device, test_loader)


    # Get some image from test dataset
    dataiter = iter(test_loader)
    print(test_loader)
    images, labels = dataiter.next()
    images, labels = Variable(images), Variable(labels)
    images, labels = images.to(device), labels.to(device)
    
    print(labels)
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(12)))

    # Feed images to Model to predict
    outputs = model(images)

    # Just get 12 result, which mean is maximun result
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(12)))


if __name__ == '__main__':
    main()
    
    


