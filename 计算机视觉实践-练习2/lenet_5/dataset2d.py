import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from matplotlib import pyplot as plt
def getdata2d():
    pipline_train = transforms.Compose([
        #随机旋转图片
        transforms.RandomHorizontalFlip(),
        #将图片尺寸resize到32x32
        transforms.Resize((32,32)),
        #将图片转化为Tensor格式
        transforms.ToTensor(),
        #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    pipline_test = transforms.Compose([
        #将图片尺寸resize到32x32
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    #下载数据集
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=pipline_train)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=pipline_test)
    #加载数据集
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    return trainloader, testloader