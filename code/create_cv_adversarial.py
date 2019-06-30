import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.attacks import CarliniWagnerL2Attack
import datetime


with open('../models/vgg_acc_85.pkl', 'rb') as f:
    net = pickle.load(f)

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=20)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=20)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

adversary = CarliniWagnerL2Attack(net, targeted=False, num_classes=10)


adversarial= dict()
adversarial['X'] = np.empty(shape=(0,3,32,32))
adversarial['label'] = np.array(())
adversarial['net_pred'] = np.array(())
adversarial['softmax_layer'] = np.empty(shape=(0,10))

for data in tqdm(testloader):
    images, labels = data
    cln_data, true_label = images.to('cuda'), labels.to('cuda')
    adv_untargeted = adversary.perturb(cln_data, true_label)
    softmax_layer = net(adv_untargeted)
    estimate_prob, estimate_class = torch.max(softmax_layer.data, 1)
    
    wrong = true_label!=estimate_class
    print(wrong)
    
    adversarial['X'] = np.concatenate((adversarial['X'],adv_untargeted[wrong].cpu()))
    adversarial['label'] = np.concatenate((adversarial['label'],labels[wrong]))
    adversarial['net_pred'] = np.concatenate((adversarial['net_pred'],estimate_class[wrong].cpu()))
    adversarial['softmax_layer'] = np.concatenate((adversarial['softmax_layer'], softmax_layer[wrong].detach().cpu()))
    

    with open('../data/adversarial_cv_2.pkl', 'wb') as f:
        pickle.dump(adversarial,f)


