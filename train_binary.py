import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import argparse
import os
import shutil
import time
##########>>>>>
import binaryconnect_clipqua  as binaryconnect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from wideresnet import WideResNet
from alexnet import AlexNet2  as AlexNet
from alexnet import AlexNet2  as AlexNet

# 添加了专用的，forward，存储权重，但没有存储norms制定（变化的）


'''# Hyper Parameters
param = {
    #'pruning_perc': 90.,
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 5000,
    'learning_rate': 0.01,
    'weight_decay': 5e-4,
}


# Data loaders
train_dataset = datasets.CIFAR10(root='../data/',train=True, download=True,
    transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset,
    batch_size=param['batch_size'], shuffle=True)

test_dataset = datasets.CIFAR10(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset,
    batch_size=param['test_batch_size'], shuffle=True)'''
# Hyper Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 128,
    'test_batch_size': 100,
    'num_epochs': 800,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
    'momentum':0.9,
}

# Data loaders
# train_dataset = datasets.CIFAR10(root='../data/',train=True, download=True,
#     transform=transforms.ToTensor())
# loader_train = torch.utils.data.DataLoader(train_dataset,
#     batch_size=param['batch_size'], shuffle=True)
#
# test_dataset = datasets.CIFAR10(root='../data/', train=False, download=True,
#     transform=transforms.ToTensor())
# loader_test = torch.utils.data.DataLoader(test_dataset,
#     batch_size=param['test_batch_size'], shuffle=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

loader_train =  torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True, pin_memory=True)

loader_test = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, pin_memory=True)


# Load the pretrained model
net = AlexNet()


if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
print("--- Pretrained network loaded ---")

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'],
#                                 weight_decay=param['weight_decay'])
optimizer = torch.optim.SGD(net.parameters(), param['learning_rate'],
                                momentum=param['momentum'],
                                weight_decay=param['weight_decay'])


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def val(model, loader):
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)
    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100. * acc,
        num_correct,
        num_samples,
    ))

    return acc

# normal training function
def train(model, loss_fn, optimizer, param, loader_train, loader_test, loader_val=None):

    model.train()
    for epoch in range(param['num_epochs']):
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

        for t, (x, y) in enumerate(loader_train):
            x_var, y_var = to_var(x), to_var(y.long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()



            optimizer.step()

        val(model, loader_test)

# train and pruning during
def train_pp(model, loss_fn, optimizer, param, loader_train, loader_test, bin_op, k=40,l=1, loader_val=None):
    #### 前面剪枝过程中，训练的SENET，然后剪完了，不再更新权重部分
    val(model, loader_test)
    model.train()
    best_accu = 0
    count_lr_m = 0
    for epoch in range(param['num_epochs']):
        model.train()

        if (epoch+1)%l==0:
            bin_op.comp_power_array()

        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

        for t, (x, y) in enumerate(loader_train):
            x_var, y_var = to_var(x), to_var(y.long())

            bin_op.binarization()

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()

            bin_op.restore()
            # model.update_grad()
            optimizer.step()

        tmp_accu = val(model, loader_test)

        if tmp_accu>best_accu:
            best_accu = tmp_accu

        print('After this pruning operation, the best accuracy is', best_accu)



        if (epoch+1)%k==0 :
            count_lr_m+=1
            print('modify learning rate')
            lr = param['learning_rate'] * (0.9 ** count_lr_m)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


## 对网络随机生成一个mask，训练
def train_rand(model, loss_fn, optimizer, param, loader_train, loader_test, ratio, loader_val=None):
    model.train()
    model.rand_mask(ratio)
    for epoch in range(param['num_epochs']):
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

        for t, (x, y) in enumerate(loader_train):
            x_var, y_var = to_var(x), to_var(y.long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (t + 1) % 100 == 0:
                print('t = %d, loss = %.8f' % (t + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        val(model, loader_test)


model = net.cuda()
bin_op=binaryconnect.BC(model)
# k控制调节lr
k=40
# l 控制调节，量化目标
l = 1

train_pp(model, criterion, optimizer, param, loader_train, loader_test, bin_op, k, l)
# train_rand(net, criterion, optimizer, param, loader_train, loader_test, 90)



torch.save(net.state_dict(), 'models/cnn_pretrained.pkl')
