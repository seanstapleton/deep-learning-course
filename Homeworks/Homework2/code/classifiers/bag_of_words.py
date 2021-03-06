import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from dataset import ReviewsDataset
from torch.utils.data import DataLoader
from bow import BagOfWordsClassifier
import pickle

def train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
   
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # If there is no training happening
    if num_epochs == 0:
        model.eval()
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)

        best_acc = running_corrects.double() / dataset_sizes['val']

    # Training for num_epochs steps
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    ####################################################################################
                    #                             END OF YOUR CODE                                     #
                    ####################################################################################
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                with open('model.pkl', 'wb') as f:
                    pickle.dump(best_model_wts, f)

                

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if num_epochs > 0:
        model.load_state_dict(best_model_wts)
    return model

def train(device, dataloaders, dataset_sizes, vocab_size):
    model = BagOfWordsClassifier(2, vocab_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.05)

    # Train the model for 25 epochs
    print('Train the model')
    model = train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
    
    train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, exp_lr_scheduler, num_epochs=0)
    return model

def main():
    datasets = {}
    datasets['train'] = ReviewsDataset('../data/train.txt')
    datasets['val'] = ReviewsDataset('../data/dev.txt', train=False, word2idx=datasets['train'].word2idx)
    
    dataset_sizes = { x: len(datasets[x]) for x in ['train', 'val'] }

    vocab_size = len(datasets['train'].word2idx)

    dataloaders = {
        x: DataLoader(datasets[x], batch_size=5, shuffle=True, num_workers=4) for x in ['train', 'val']
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = train(device, dataloaders, dataset_sizes, vocab_size)

if __name__== "__main__":
    main()
