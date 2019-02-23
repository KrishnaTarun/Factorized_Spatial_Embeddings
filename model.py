from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn


class ConvNet(nn.Module):

  def __init__(self, n_channels, n_classes):
    super(ConvNet,self).__init__()

    self.conv1 = nn.Conv2d(n_channels,20,5,stride=1,padding=0,bias=False)
    self.batchnorm1 = nn.BatchNorm2d(20,momentum=0.9)
    self.maxpool1 = nn.MaxPool2d(2,stride=2)
    
    self.conv2 = nn.Conv2d(20,48,5,stride=1,padding=0,bias=False)
    self.batchnorm2 = nn.BatchNorm2d(48,momentum=0.9)
    
    self.conv3 = nn.Conv2d(48,64,5,stride=1,padding=0,bias=False)
    self.batchnorm3 = nn.BatchNorm2d(64,momentum=0.9)
    
    self.conv4 = nn.Conv2d(64,80,5,stride=1,padding=0,bias=False)
    self.batchnorm4 = nn.BatchNorm2d(80,momentum=0.9)
    
    self.conv5 = nn.Conv2d(80,256,5,stride=1,padding=0,bias=False)
    self.batchnorm5 = nn.BatchNorm2d(256, momentum=0.9)
    
    self.conv6 = nn.Conv2d(256,n_classes,5,stride=1,padding=0,bias=False)
    self.batchnorm6 = nn.BatchNorm2d(n_classes, momentum=0.9)
    


    self.ReLU = nn.ReLU()
    

    #TODO normal weight initializer

  def init_w(self):
    for conv in self.layers:
      nn.init.normal_(conv.weight,0,0.02)

  def forward(self, x):

    out = self.ReLU(self.batchnorm1(self.conv1(x)))
    out = self.maxpool1(out)
    out = self.ReLU(self.batchnorm2(self.conv2(out)))
    
    out = self.ReLU(self.batchnorm3(self.conv3(out)))
    out = self.ReLU(self.batchnorm4(self.conv4(out)))
    out = self.ReLU(self.batchnorm5(self.conv5(out)))
    out = self.ReLU(self.batchnorm6(self.conv6(out)))
    
    return out
  