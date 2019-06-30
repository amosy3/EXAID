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
from advertorch.attacks import CarliniWagnerL2Attack, PGDAttack, FGSM, JSMA
import datetime
import argparse

def get_test_loader(dataset):
    if dataset == 'MNIST':
        pass

    if dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=20)
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return testloader
    print('Testloader error!')


def get_pretrain_model(model_name):
    if model_name == 'vgg':
        model_path = '../models/vgg_acc_85.pkl'

    if model_name == 'googlenet':
        model_path = '../models/googlenet_acc_84.pkl'

    if model_name == 'resnet':
        model_path = '../models/resnet_acc_90.pkl'

    with open(model_path, 'rb') as f:
        net = pickle.load(f)
    return net


def get_adversary(adversary_name, net):
    if adversary_name == 'FGSM':
        adversary = FGSM(net, targeted=False, num_classes=10)

    if adversary_name == 'JSMA':
        adversary = JSMA(net, targeted=False, num_classes=10)

    if adversary_name == 'PGD':
        adversary = PGDAttack(net, targeted=False, num_classes=10)

    if adversary_name == 'CW':
        adversary = CarliniWagnerL2Attack(net, targeted=False, num_classes=10)

    return adversary


    with open(model_path, 'rb') as f:
        net = pickle.load(f)
    return net

def print_net_score(net, testloader):
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

    print('Accuracy of the network on test set: %d %%' % (100 * correct / total))


parser = argparse.ArgumentParser(description='This script create adversarial examples. '
                                             'Please choose dataset, model, and attack to apply')
parser.add_argument('dataset', action='store',choices=['MNIST','CIFAR10'], type=str, help='dataset')
parser.add_argument('model', action='store',choices=['resnet', 'vgg', 'googlenet'], type=str, help='model')
parser.add_argument('attack', action='store',choices=['FGSM', 'JSMA', 'PGD', 'CW'], type=str, help='attack')
args = parser.parse_args()

testloader = get_test_loader(args.dataset)
net = get_pretrain_model(args.model)
adversary = get_adversary(args.attack, net)

print_net_score(net, testloader)

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
    

adversarial_path = '../data/%s/%s/%s.pkl' %(args.dataset, args.model, args.attack)
with open(adversarial_path, 'wb') as f:
    pickle.dump(adversarial,f)