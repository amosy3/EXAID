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


def get_test_loader(dataset, batch_size=100, num_workers=35):
    if dataset == 'MNIST':
        pass

    if dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return testloader

    if dataset == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return testloader

    if dataset == 'ImageNet':
        # no labels on validation... using train instead
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), normalize])
        train_set = torchvision.datasets.ImageFolder('../data/ImageNet/train/', transform=transform)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers)

        return trainloader
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
            net.eval()
            net = net.to('cuda')
            return net

        if args.dataset == 'ImageNet':
            net = torchvision.models.resnet152(pretrained=True)
            net.eval()
            net = net.to('cuda')
            return net

    with open(model_path, 'rb') as f:
        net = pickle.load(f)
    return net.module


def get_adversary(args, net):

    if args.dataset == 'IMAGENET':
        num_classes = 1000
    else:
        num_classes = 10


    if args.attack == 'FGSM':
        adversary = FGSM(net, eps=args.attack_eps, targeted=False)

    if args.attack == 'JSMA':
        adversary = JSMA(net, num_classes=num_classes)

    if args.attack == 'PGD':
        adversary = PGDAttack(net, eps=args.attack_eps, targeted=False)

    if args.attack == 'CW':
        adversary = CarliniWagnerL2Attack(net, initial_const=args.attack_eps, targeted=False, num_classes=num_classes)

    return adversary


    with open(model_path, 'rb') as f:
        net = pickle.load(f)
    return net


def config_gpus_setup(model, gpus='all'):
    """
    :param model: pytorch models
    :param gpus: list of gpus ids to use or 'all' to use all of them (default)
    :return:
    """

    ngpus = torch.cuda.device_count()
    print('Cuda see %s GPUs' % ngpus)

    if gpus == 'all':
        gpus = list(range(ngpus))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        if (ngpus > 1):
            model = torch.nn.DataParallel(model, device_ids=gpus)
    else:
        device = torch.device("cpu")

    return model, device


def print_net_score(net, testloader):

    # net, _ = config_gpus_setup(net)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test set: %d %%' % (100 * correct / total))

    del images
    torch.cuda.empty_cache()


def get_parsed_args():
    parser = argparse.ArgumentParser(description='This script create adversarial examples. '
                                                 'Please choose dataset, model, and attack to apply')
    parser.add_argument('dataset', action='store',choices=['MNIST','CIFAR10', 'SVHN','ImageNet'], type=str, help='dataset')
    parser.add_argument('model', action='store',choices=['resnet', 'vgg', 'googlenet'], type=str, help='model')
    parser.add_argument('attack', action='store',choices=['FGSM', 'JSMA', 'PGD', 'CW'], type=str, help='attack')
    parser.add_argument('attack_eps', action='store', type=float, help='FGSM/PGD step size, or CW l2 weight in loss')
    parser.add_argument('--batch_size', action='store', type=int, help='Config batch size')
    args = parser.parse_args()
    return args


def print_cuda_stats(ids='all'):
    if ids == 'all':
        gpus = list(range(torch.cuda.device_count()))
    print('Memory status for GPUs:')
    for j in gpus:
        alloc = torch.cuda.memory_allocated('cuda:%s' % j) / 1000000000
        cached = torch.cuda.memory_cached('cuda:%s' % j) / 1000000000
        print('GPU - %s: alloc: %s G/ cached: %s G' %(j, alloc, cached))

args = get_parsed_args()
testloader = get_test_loader(args.dataset, args.batch_size)
net = get_pretrain_model(args)
net, _ = config_gpus_setup(net)
adversary = get_adversary(args, net)

# print_net_score(net, testloader)  # ensure the net loaded as expected



# if args.dataset == 'IMAGENET':
#     adversarial['X'] = np.empty(shape=(0, 3, 224, 224))
#     adversarial['softmax_layer'] = np.empty(shape=(0, 1000))
# else:
#     adversarial['X'] = np.empty(shape=(0, 3, 32, 32))
#     adversarial['softmax_layer'] = np.empty(shape=(0, 10))
# adversarial['label'] = np.array(())
# adversarial['net_pred'] = np.array(())

adversarial = dict()

for i, data in enumerate(tqdm(testloader)):
    images, labels = data
    cln_data, true_label = images.to('cuda'), labels.to('cuda')
    adv_untargeted = adversary.perturb(cln_data, true_label)
    softmax_layer = net(adv_untargeted)
    estimate_prob, estimate_class = torch.max(softmax_layer.data, 1)
    
    wrong = true_label != estimate_class
    print('successful attacks: %s/%s (%0.2f)' %(wrong.sum().item(), args.batch_size, 100*wrong.sum()/args.batch_size))

    # adversarial['X'] = np.concatenate((adversarial['X'],adv_untargeted[wrong].detach().cpu()))
    # adversarial['label'] = np.concatenate((adversarial['label'],labels[wrong]))
    # adversarial['net_pred'] = np.concatenate((adversarial['net_pred'],estimate_class[wrong].cpu()))
    # adversarial['softmax_layer'] = np.concatenate((adversarial['softmax_layer'], softmax_layer[wrong].detach().cpu()))

    adversarial['X'] = adv_untargeted[wrong].detach().cpu()
    adversarial['label'] = labels[wrong]
    adversarial['net_pred'] = estimate_class[wrong].cpu()
    adversarial['softmax_layer'] = softmax_layer[wrong].detach().cpu()
    

    adversarial_path = '../adversarial/%s/%s/' %(args.dataset, args.model)
    create_dir_if_not_exist(adversarial_path)
    with open(adversarial_path + args.attack + '_' + str(args.attack_eps) + '_%s.pkl'%i, 'wb') as f:
        pickle.dump(adversarial,f)

    del images, cln_data, adv_untargeted, softmax_layer#, estimate_prob, estimate_class, true_label
    torch.cuda.empty_cache()


    if i>500:
        print('You got %s adversarial examples - that should do the job...'%(i*args.batch_size))
        break
