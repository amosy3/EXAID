import torch, torchvision
from torchvision import datasets, transforms

from torch import nn, optim
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np
import pickle
import shap

with open('../models/vgg_acc_85.pkl', 'rb') as f:
    net = pickle.load(f)
net.eval()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=20)

batch = next(iter(trainloader))
images, labels = batch

background = images[:100]
e = shap.DeepExplainer(net, background.to('cuda'))

def get_xai_channels(X,e):
    data = (torch.from_numpy(X)).to('cuda')
    shap_values = e.shap_values(data.float())
    return np.transpose(np.array(shap_values),(1,0,2,3,4))

filenames = ['natural_misclassified']

print('Start working :-)')

for filename in filenames:
    with open('../data/'+ filename +'.pkl', 'rb') as f:
        data = pickle.load(f)
    data['shap'] = get_xai_channels(data['X'],e)
    with open('../data/final_db/'+ filename +'.pkl', 'wb') as f:
        pickle.dump(data, f)
    print('done!')