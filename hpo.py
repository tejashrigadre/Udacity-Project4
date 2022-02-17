#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import ImageFile


import json
import logging
import os
import sys
import argparse
from smdebug.profiler.utils import str2bool

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def train(model, train_loader, valid_loader, criterion, optimizer, device, args):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    image_dataset = {'train' : train_loader, 'valid' : valid_loader}
    batch_size = args.batch_size
    epoch = args.epochs
    best_loss=1e6
    loss_counter = 0


    
    for i in range(epoch):  
        for phase in ['train', 'valid']:
            #print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                start = time.time
                model.train()
                                   
            if phase=='valid':
                  start = time.time
                  model.eval()
            
                 
            running_loss = 0.0
            running_corrects = 0
            running_samples=0
          
            for inputs, labels in image_dataset[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                
               
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                
                
                epoch_loss = running_loss / len(inputs)
                epoch_acc = running_corrects / len(inputs) 
                running_samples+=len(inputs)
               
                if phase=='valid':
                    logger.info("validating the model")
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
                 
                logger.info('{} loss: {}, acc: {}'.format(phase,epoch_loss,epoch_acc))
                logger.info(f"{phase} Accuracy: {100*epoch_acc}, {phase} Loss: {epoch_loss}")
                            
        if loss_counter == 1:
            break
           
      
    return model   
            

def test(model, test_loader, criterion, device, args):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    test_loss = 0
    correct = 0
    model.eval()
        
        
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)                  
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        test_loss += loss.item() * inputs.size(0)
        correct += torch.sum(preds == labels.data).item()

    total_loss = test_loss / len(test_loader.dataset)
    total_acc = correct/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset))
    )
    

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
            nn.Linear(num_features,133)
        )
    return model

def create_data_loaders(train, valid, test, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Get train data loader")
    #batch_size = {"batch_size": args.batch_size}
        
    #train_data_path = os.path.join(train, 'train')
    #print(train_data_path)
    #valid_data_path = os.path.join(valid, 'valid')
    #test_data_path = os.path.join(test, 'test')
    train_data_path  = train
    valid_data_path  = valid
    test_data_path  = test
    
 
    
    transform_train = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    transform_valid = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    dataset_train = datasets.ImageFolder(train_data_path, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        
    dataset_valid = datasets.ImageFolder(valid_data_path, transform=transform_valid)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size)
        
    dataset_test = datasets.ImageFolder(test_data_path, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size )

    print(f"Train data path  {train_data_path}")
    print(f"Test data path  {test_data_path}")
    print(f"Valid data path  {valid_data_path}")
    
    return train_loader, valid_loader, test_loader
    
    

def main(args):
    
    #if args.gpu == 1:
    #    device = torch.device("cuda")
    #else:
     #   device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    logger.info(f"Running on Device {device}")
    train_loader, valid_loader, test_loader = create_data_loaders(args.train, args.valid, args.test, args.batch_size)
    
    '''
    TODO: Initialize a model by calling the net function
    '''
        
    model=net()
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Training the model")
    model=train(model, train_loader, valid_loader, criterion, optimizer,device, args)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing the model")
    test(model, test_loader, criterion,device, args)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model")
    path = os.path.join(args.modeldir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help = "Input batch size for training(default : 64)"
    )
        
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help = "Number of epochs(default : 14)"
    )    
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help = "Learning Rate : (default:1.0)"
    )
            

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--modeldir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--valid', type=str, default=os.environ['SM_CHANNEL_VALID'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)

