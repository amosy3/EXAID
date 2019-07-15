import torch, torchvision
from torchvision import datasets, transforms
import os
import argparse
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shap


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_parsed_args():
    parser = argparse.ArgumentParser(description='This script creates explanations for net predictions based on shap. '
                                                 'Please choose dataset for natural samples, model, '
                                                 'and attack to explain')
    parser.add_argument('dataset', action='store',choices=['MNIST','CIFAR10'], type=str, help='dataset')
    parser.add_argument('model', action='store',choices=['resnet', 'vgg', 'googlenet'], type=str, help='model')
    parser.add_argument('attack', action='store',choices=['FGSM', 'JSMA', 'PGD', 'CW'], type=str, help='attack')
    args = parser.parse_args()
    return args


def get_loaders(dataset):
    if dataset == 'MNIST':
        pass

    if dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=20)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=20)
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return trainloader, testloader
    print('loaders error!')


def get_pretrain_model(model_name):
    if model_name == 'vgg':
        model_path = '../models/vgg_acc_85.pkl'

    if model_name == 'googlenet':
        model_path = '../models/googlenet_acc_84.pkl'

    if model_name == 'resnet':
        model_path = '../models/resnetxt_acc_87.pkl'

    with open(model_path, 'rb') as f:
        net = pickle.load(f)
    return net.module


def get_adversarial(dataset,model,attack):
    adversarial_path = '../adversarial/%s/%s/%s.pkl' % (dataset, model, attack)
    with open(adversarial_path, 'rb') as f:
        adversarial_sampels = pickle.load(f)
    return adversarial_sampels


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


def get_xai_channels(X,e):
    if type(X).__module__ == np.__name__:
        data = (torch.from_numpy(X)).to('cuda')
    else:
        data = X
    shap_values = e.shap_values(data.float())
    return np.transpose(np.array(shap_values),(1,0,2,3,4))


def get_deep_explainer(trainloader, net):
    batch = next(iter(trainloader))
    images, labels = batch
    background = images[:100]
    e = shap.DeepExplainer(net, background.to('cuda'))
    return e


def create_adversarial_explanations(adversarial, e, dataset, model, attack):
    adversarial['shap'] = get_xai_channels(adversarial['X'], e)
    with open('../explanations/%s/%s/%s.pkl' % (dataset, model, attack), 'wb') as f:
        pickle.dump(adversarial, f)


def create_natural_explanations(trainloader, e, dataset, model):
    natural = dict()
    if dataset == 'MNIST':
        natural['X'] = np.empty(shape=(0, 1, 28, 28))
        natural['shap'] = np.empty(shape=(0, 10, 1, 28, 28))
    if dataset == 'CIFAR10':
        natural['X'] = np.empty(shape=(0, 3, 32, 32))
        natural['shap'] = np.empty(shape=(0, 10, 3, 32, 32))
    natural['label'] = np.array(())
    natural['net_pred'] = np.array(())
    natural['softmax_layer'] = np.empty(shape=(0, 10))

    for data in tqdm(trainloader):
        images, labels = data
        images = images.to('cuda')
        softmax_layer = net(images)
        estimate_prob, estimate_class = torch.max(softmax_layer.data, 1)
        shap_vaules = get_xai_channels(images, e)

        natural['X'] = np.concatenate((natural['X'], images.cpu()))
        natural['label'] = np.concatenate((natural['label'], labels))
        natural['net_pred'] = np.concatenate((natural['net_pred'], estimate_class.cpu()))
        natural['softmax_layer'] = np.concatenate((natural['softmax_layer'], softmax_layer.detach().cpu()))
        natural['shap'] = np.concatenate((natural['shap'], shap_vaules))
    with open('../explanations/%s/%s/natural.pkl' % (dataset, model), 'wb') as f:
        pickle.dump(natural, f)

args = get_parsed_args()
trainloader, testloader = get_loaders(args.dataset)
net = get_pretrain_model(args.model)
adversarial = get_adversarial(args.dataset,args.model,args.attack)
print_net_score(net, testloader)  # ensure the net loaded as expected
e = get_deep_explainer(trainloader, net)
create_adversarial_explanations(adversarial, e, args.dataset,args.model,args.attack)
print('Done explain adversarial examples!')

if not os.path.exists('../explanations/%s/%s/natural.pkl' % (args.dataset, args.model)):
    create_natural_explanations(testloader, e, args.dataset, args.model)
