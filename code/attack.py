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
from models import mahalanobis_resnet as mres
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.attacks import CarliniWagnerL2Attack, PGDAttack, FGSM, JSMA
import datetime
import argparse
import os


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_test_loader(dataset):
    if dataset == 'MNIST':
        pass

    if dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=20)
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return testloader

    if dataset == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=20)
        return testloader
    print('Testloader error!')


def get_pretrain_model(args):
    model_name = args.model
    if model_name == 'vgg':
        model_path = '../models/vgg_acc_85.pkl'

    if model_name == 'googlenet':
        model_path = '../models/googlenet_acc_84.pkl'

    if model_name == 'resnet':
        if args.dataset == 'CIFAR10':
            model_path = '../models/resnetxt_acc_87.pkl'

        if args.dataset == 'SVHN':
            net = mres.ResNet34(num_c=10)
            net.load_state_dict(torch.load('../models/resnet_svhn.pth'))
            net = net.to('cuda')
            return net

    with open(model_path, 'rb') as f:
        net = pickle.load(f)
    return net.module


def get_adversary(args, net):
    if args.attack == 'FGSM':
        adversary = FGSM(net, eps=args.attack_eps, targeted=False)

    if args.attack == 'JSMA':
        adversary = JSMA(net, num_classes=10)

    if args.attack == 'PGD':
        adversary = PGDAttack(net, eps=args.attack_eps, targeted=False)

    if args.attack == 'CW':
        adversary = CarliniWagnerL2Attack(net, initial_const=args.attack_eps, targeted=False, num_classes=10)

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


def get_parsed_args():
    parser = argparse.ArgumentParser(description='This script create adversarial examples. '
                                                 'Please choose dataset, model, and attack to apply')
    parser.add_argument('dataset', action='store',choices=['MNIST','CIFAR10', 'SVHN'], type=str, help='dataset')
    parser.add_argument('model', action='store',choices=['resnet', 'vgg', 'googlenet'], type=str, help='model')
    parser.add_argument('attack', action='store',choices=['FGSM', 'JSMA', 'PGD', 'CW'], type=str, help='attack')
    parser.add_argument('attack_eps', action='store', type=float, help='FGSM/PGD step size, or CW l2 weight in loss')
    args = parser.parse_args()
    return args


args = get_parsed_args()
testloader = get_test_loader(args.dataset)
net = get_pretrain_model(args)
adversary = get_adversary(args, net)

print_net_score(net, testloader)  # ensure the net loaded as expected

adversarial= dict()
adversarial['X'] = np.empty(shape=(0, 3, 32, 32))
adversarial['label'] = np.array(())
adversarial['net_pred'] = np.array(())
adversarial['softmax_layer'] = np.empty(shape=(0, 10))

for data in tqdm(testloader):
    images, labels = data
    cln_data, true_label = images.to('cuda'), labels.to('cuda')
    adv_untargeted = adversary.perturb(cln_data, true_label)
    softmax_layer = net(adv_untargeted)
    estimate_prob, estimate_class = torch.max(softmax_layer.data, 1)
    
    wrong = true_label!=estimate_class
    print(wrong)
    
    adversarial['X'] = np.concatenate((adversarial['X'],adv_untargeted[wrong].detach().cpu()))
    adversarial['label'] = np.concatenate((adversarial['label'],labels[wrong]))
    adversarial['net_pred'] = np.concatenate((adversarial['net_pred'],estimate_class[wrong].cpu()))
    adversarial['softmax_layer'] = np.concatenate((adversarial['softmax_layer'], softmax_layer[wrong].detach().cpu()))
    

    adversarial_path = '../adversarial/%s/%s/' %(args.dataset, args.model)
    create_dir_if_not_exist(adversarial_path)
    with open(adversarial_path + args.attack + '_' + str(args.attack_eps) + '.pkl', 'wb') as f:
        pickle.dump(adversarial,f)

    if adversarial['X'].shape[0]>5000:
        print('You got 5000 adversarial examples - that should do the job...')
        break
