import os
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import set_seed, write_config_log, write_result_log

import config as cfg
import numpy as np

def plot_learning_curve(logfile_dir, result_lists):
    filepath=os.path.join(logfile_dir, 'result_log.txt')
    f= open(filepath, 'r')
    train_acc=[]
    train_loss=[]
    val_acc=[]
    val_loss=[]
    epo=0
    for line in f.readlines():
        epo+=1
        train_acc.append(float(line[line.find('Train Acc')+11:line.find('Train Acc')+18]))
        train_loss.append(float(line[line.find('Train Loss')+12:line.find('Train Loss')+19]))
        val_acc.append(float(line[line.find('Val Acc')+9:line.find('Val Acc')+17]))
        val_loss.append(float(line[line.find('Val Loss')+10:line.find('Val Loss')+17]))
    epoch=np.arange(1,epo+1)

    ## ploy Train Accuracy
    fig = plt.figure()
    f1 = np.polyfit(epoch, train_acc, 5)
    p1 = np.poly1d(f1)
    yvals1=p1(epoch)
    plt.plot(epoch, train_acc, 'ro', label='train acc')
    plt.plot(epoch, yvals1, 'r',label='polyfit train acc')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.legend()
    if cfg.model_type == 'mynet':
        plt.savefig('mynet_Train_Accuracy.png')
        # print(model.state_dict())
    elif cfg.model_type == 'resnet18':
        plt.savefig('ResNet18_Train_Accuracy.png')
    ## ploy Train_loss
    fig = plt.figure()
    f1 = np.polyfit(epoch, train_loss, 5)
    p1 = np.poly1d(f1)
    yvals1=p1(epoch)
    plt.plot(epoch, train_loss, 'ro', label='train loss')
    plt.plot(epoch, yvals1, 'r',label='polyfit train loss')
    plt.xlabel('epoch')
    plt.ylabel('Error')
    plt.title('Train loss')
    plt.legend()
    if cfg.model_type == 'mynet':
        plt.savefig('mynet_Train_loss.png')
    elif cfg.model_type == 'resnet18':
        plt.savefig('ResNet18_Train_loss.png')
    
    ## plot Val_Accuracy
    fig = plt.figure()
    f1 = np.polyfit(epoch, val_acc, 5)
    p1 = np.poly1d(f1)
    yvals1=p1(epoch)
    plt.plot(epoch, val_acc, 'ro', label='Val Accuracy')
    plt.plot(epoch, yvals1, 'r',label='polyfit Val Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Val Accuracy')
    plt.legend()
    if cfg.model_type == 'mynet':
        plt.savefig('mynet_Val_Accuracy.png')
    elif cfg.model_type == 'resnet18':
        plt.savefig('ResNet18_Val_Accuracy.png')

    ## plot Val_Loss
    fig = plt.figure()
    f1 = np.polyfit(epoch, val_loss, 5)
    p1 = np.poly1d(f1)
    yvals1=p1(epoch)
    plt.plot(epoch, val_loss, 'ro', label='Val Loss')
    plt.plot(epoch, yvals1, 'r',label='polyfit Val Loss')
    plt.xlabel('epoch')
    plt.ylabel('Error')
    plt.title('Val Loss')
    plt.legend()
    plt.tight_layout() 
    if cfg.model_type == 'mynet':
        plt.savefig('mynet_Val_Loss.png')
        # print(model.state_dict())
    elif cfg.model_type == 'resnet18':
        plt.savefig('ResNet18_Val_Loss.png')


def train(model, train_loader, val_loader, logfile_dir, model_save_dir, criterion, optimizer, scheduler, device):

    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        ##### TRAINING #####
        train_start_time = time.time()
        train_loss = 0.0
        train_correct = 0.0
        model.train()
        for batch, data in enumerate(train_loader):
            sys.stdout.write(f'\r[{epoch + 1}/{cfg.epochs}] Train batch: {batch + 1} / {len(train_loader)}')
            sys.stdout.flush()
            # Data loading.
            images, labels = data['images'].to(device), data['labels'].to(device) # (batch_size, 3, 32, 32), (batch_size)
            # Forward pass. input: (batch_size, 3, 32, 32), output: (batch_size, 10)
            pred = model(images)
            # Calculate loss.
            loss = criterion(pred, labels)
            # Backprop. (update model parameters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Evaluate.
            train_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
            train_loss += loss.item()
        # Print training result
        train_time = time.time() - train_start_time
        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        print()
        print(f'[{epoch + 1}/{cfg.epochs}] {train_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Train Loss: {train_loss:.5f}')

        ##### VALIDATION #####
        model.eval()
        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0.0
            val_correct = 0.0
            for  batch,data in enumerate(val_loader):
                images, labels = data['images'].to(device), data['labels'].to(device)
                
                val = model(images)
                loss = criterion(val,labels)
                val_correct += torch.sum(torch.argmax(val, dim=1) == labels)
                val_loss += loss.item()

        # Print validation result
        val_time = time.time() - val_start_time
        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        print()
        print(f'[{epoch + 1}/{cfg.epochs}] {val_time:.2f} sec(s) Val Acc: {val_acc:.5f} | Val Loss: {val_loss:.5f}')
        
        # Scheduler step
        scheduler.step()

        ##### WRITE LOG #####
        is_better = val_acc >= best_acc
        epoch_time = train_time + val_time
        write_result_log(os.path.join(logfile_dir, 'result_log.txt'), epoch, epoch_time, train_acc, val_acc, train_loss, val_loss, is_better)

        ##### SAVE THE BEST MODEL #####
        if is_better:
            if cfg.model_type == 'mynet':
                print(f'[{epoch + 1}/{cfg.epochs}] Save best model to {model_save_dir} ...')
                torch.save(model.state_dict(), os.path.join(model_save_dir, 'mynet_best.pth'))
                best_acc = val_acc
            elif cfg.model_type == 'resnet18':
                torch.save(model.state_dict(), os.path.join(model_save_dir, 'resnet18_best.pth'))
                best_acc = val_acc
        ##### PLOT LEARNING CURVE #####
        ##### TODO: check plot_learning_curve() in this file #####
        current_result_lists = {
            'train_acc': train_acc_list,
            'train_loss': train_loss_list,
            'val_acc': val_acc_list,
            'val_loss': val_loss_list
        }
        plot_learning_curve(logfile_dir, current_result_lists)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='dataset directory', type=str, default='../hw2_data/p2_data/')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    # Experiment name
    exp_name = cfg.model_type + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S') + '_' + cfg.exp_name

    # Write log file for config
    logfile_dir = os.path.join('./experiment', exp_name, 'log')
    os.makedirs(logfile_dir, exist_ok=True)
    write_config_log(os.path.join(logfile_dir, 'config_log.txt'))

    # Fix a random seed for reproducibility
    set_seed(9527)

    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Device:', device)

    ##### MODEL #####
    ##### TODO: check model.py #####
    model_save_dir = os.path.join('./experiment', exp_name, 'model')
    os.makedirs(model_save_dir, exist_ok=True)

    if cfg.model_type == 'mynet':
        model = MyNet()
    elif cfg.model_type == 'resnet18':
        model = ResNet18()
    else:
        raise NameError('Unknown model type')

    model.to(device)

    ##### DATALOADER #####
    ##### TODO: check dataset.py #####
    train_loader = get_dataloader(os.path.join(dataset_dir, 'train'), batch_size=cfg.batch_size, split='train')
    val_loader   = get_dataloader(os.path.join(dataset_dir, 'val'), batch_size=cfg.batch_size, split='val')

    ##### LOSS & OPTIMIZER #####
    criterion = nn.CrossEntropyLoss()
    if cfg.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.1)
    
    ##### TRAINING & VALIDATION #####
    ##### TODO: check train() in this file #####
    train(model          = model,
          train_loader   = train_loader,
          val_loader     = val_loader,
          logfile_dir    = logfile_dir,
          model_save_dir = model_save_dir,
          criterion      = criterion,
          optimizer      = optimizer,
          scheduler      = scheduler,
          device         = device)
    
if __name__ == '__main__':
    main()
