# import sys
# import matplotlib.mlab as mlab
# import scipy.integrate as integrate
# from PIL import Image
# from scipy import fft
# from scipy import misc
# from skimage import transform
# import shutil
# # import requests
# import tempfile
# import os
from models import mahalanobis_resnet as mres
import math
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch, torchvision
from torchvision import datasets, transforms
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import argparse
import numpy as np
from hyperopt import hp, tpe, fmin


def get_parsed_args():
    parser = argparse.ArgumentParser(description='This script creates ANR detection for adversarial samples.'
                                                 'Include internal hyperopt to fit the dataset ')
    parser.add_argument('dataset', action='store',choices=['MNIST','CIFAR10', 'SVHN'], type=str, help='dataset')
    parser.add_argument('model', action='store',choices=['resnet', 'vgg', 'googlenet'], type=str, help='model')
    parser.add_argument('nsamples2hpo', action='store', type=int, help='how many samples to use for hyperparameter'
                                                                       ' optimization')
    parser.add_argument('hpo_steps', action='store', type=int, help='how many epoches to use for hyperparameter'
                                                                       ' optimization')
    args = parser.parse_args()
    return args


def oneDEntropy(inputDigit):
    inputDigit = (inputDigit.numpy()*255).astype(int)
    expandDigit = np.array(inputDigit,dtype=np.int16)
    f1 = np.zeros(256)
    f2 = np.zeros(256)
    f3 = np.zeros(256)
    for i in range(32):
        for j in range(32):
            f1[expandDigit[0][i][j]]+=1
            f2[expandDigit[1][i][j]]+=1
            f3[expandDigit[2][i][j]]+=1
    f1/=1024.0
    f2/=1024.0
    f3/=1024.0
    H1 = 0
    H2 = 0
    H3 = 0
    for i in range(256):
        if f1[i] > 0:
            H1+=f1[i]*math.log(f1[i],2)
        if f2[i] > 0:
            H2+=f2[i]*math.log(f2[i],2)
        if f3[i] > 0:
            H3+=f3[i]*math.log(f3[i],2)
    return -(H1+H2+H3)/3.0


def scalarQuantization(inputDigit, interval):
    retDigit = np.array(inputDigit,dtype=np.float32)
    retDigit *= 255
    retDigit//=interval
    retDigit*=interval
    return retDigit


def crossMeanFilterOperations(inputDigit, start, end, coefficient):
    retDigit = np.array(inputDigit, dtype=np.float32)

    for row in range(start, end):
        for col in range(start, end):
            temp0 = inputDigit[0][row][col]
            temp1 = inputDigit[1][row][col]
            temp2 = inputDigit[2][row][col]
            for i in range(1,start+1):
                temp0+=inputDigit[0][row-i][col]
                temp0+=inputDigit[0][row+i][col]
                temp0+=inputDigit[0][row][col-i]
                temp0+=inputDigit[0][row][col+i]
                temp1+=inputDigit[1][row-i][col]
                temp1+=inputDigit[1][row+i][col]
                temp1+=inputDigit[1][row][col-i]
                temp1+=inputDigit[1][row][col+i]
                temp2+=inputDigit[2][row-i][col]
                temp2+=inputDigit[2][row+i][col]
                temp2+=inputDigit[2][row][col-i]
                temp2+=inputDigit[2][row][col+i]
            retDigit[0][row][col]= temp0/coefficient
            retDigit[1][row][col] = temp1/coefficient
            retDigit[2][row][col] = temp2/coefficient
    return retDigit


def chooseCloserFilter(original_data,filter_data1,filter_data2):
    result_data=np.zeros_like(original_data)
    for j in range(32):
        for k in range(32):
            for i in range(3):
                a=abs(filter_data1[i][j][k]-original_data[i][j][k])
                b=abs(filter_data2[i][j][k]-original_data[i][j][k])
                if(a<b):
                    result_data[i][j][k]=filter_data1[i][j][k]
                else:
                    result_data[i][j][k]=filter_data2[i][j][k]
    return result_data


def load_data(dataset,model):

    with open('../adversarial/%s/%s/FGSM.pkl' %(dataset,model), 'rb') as f:
        adversarial_FGSM = pickle.load(f)

    with open('../adversarial/%s/%s/PGD.pkl' %(dataset,model), 'rb') as f:
        adversarial_PGD = pickle.load(f)

    with open('../adversarial/%s/%s/CW.pkl' %(dataset,model), 'rb') as f:
        adversarial_CW = pickle.load(f)

    return adversarial_FGSM, adversarial_PGD, adversarial_CW


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


def numpy2torch(x):
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    return torch.tensor(x).float().to('cuda')


def detect(data, q1=128, q2=64, q3=43, en1=4, en2=1):
    detection_pred = []
    for image in data:
        raw_pred = torch.argmax(model(numpy2torch(image)), dim=1)
        ori_entropy = oneDEntropy(image)
        #         print(ori_entropy)
        #         image = np.transpose(image,(1,2,0))
        if ori_entropy < en1:
            ppimage = scalarQuantization(image, q1)

        elif ori_entropy < (en1+en2):
            ppimage = scalarQuantization(image, q2)

        else:
            qimage = scalarQuantization(image, q3)
            fqimage = crossMeanFilterOperations(qimage, 3, 29, 13)
            ppimage = chooseCloserFilter(image, qimage, fqimage)

        pp_pred = torch.argmax(model(numpy2torch(ppimage)), dim=1)

        if raw_pred == pp_pred:
            detection_pred.append(0)
        else:
            detection_pred.append(1)

    return detection_pred


args = get_parsed_args()

model = get_pretrain_model(args)
testloader = get_test_loader(args.dataset)

with open('../adversarial/%s/%s/FGSM_0.1.pkl' % (args.dataset, args.model), 'rb') as f:
    adversarial_FGSM = pickle.load(f)

natural = np.empty(shape=(0, 3, 32, 32))
for data in (testloader):
    images, labels = data
    natural = np.concatenate((natural, images))
natural_train = natural[:args.nsamples2hpo]
natural_test = natural[args.nsamples2hpo:]


# detect_natural = []
# for data,label in tqdm(testloader):
#     detect_natural += detect(data)
# detect_natural_train = detect_natural[:args.nsamples2hpo]
# detect_natural_test = detect_natural[args.nsamples2hpo:]

def objective(params):
    print(params)
    X = torch.tensor(np.concatenate((natural_train,adversarial_FGSM['X'][:args.nsamples2hpo])))
    y = [0] * args.nsamples2hpo + [1] * args.nsamples2hpo
    detect_preds = detect(X, params['q1'], params['q2'], params['q3'], params['en1'], params['en2'])

    return 1 - roc_auc_score(y, detect_preds)

# best = {'q1':128, 'q2':64, 'q3':43, 'en1':4, 'en2':1} #default values from original git repo

space = {
    'q1': hp.choice('q1', range(100)),
    'q2': hp.choice('q2', range(100)),
    'q3': hp.choice('q3', range(100)),
    'en1': hp.choice('en1', range(10)),
    'en2': hp.choice('en2', range(5))
}

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=args.hpo_steps)

nval = 5*args.nsamples2hpo

print('FGSM')
for e in [0.3,0.1,0.03,0.01,0.003,0.001,0.0003]:
    with open('../adversarial/%s/%s/FGSM_%s.pkl' %(args.dataset, args.model,e), 'rb') as f:
        adversarial = pickle.load(f)
    X = torch.tensor(np.concatenate((natural_test, adversarial['X'][:nval])))
    y = [0]*len(natural_test) + [1]*adversarial['X'][:nval].shape[0]
    detect_pred = detect(X,best['q1'],best['q2'],best['q3'],best['en1'],best['en2'])
    print(e,roc_auc_score(y,detect_pred))

print('PGD')
for e in [0.3,0.1,0.03,0.01,0.003,0.001,0.0003]:
    with open('../adversarial/%s/%s/PGD_%s.pkl' %(args.dataset, args.model,e), 'rb') as f:
        adversarial = pickle.load(f)
    X = torch.tensor(np.concatenate((natural_test, adversarial['X'][:nval])))
    y = [0]*len(natural_test) + [1]*adversarial['X'][:nval].shape[0]
    detect_pred = detect(X,best['q1'],best['q2'],best['q3'],best['en1'],best['en2'])
    print(e,roc_auc_score(y,detect_pred))

print('CW')
for e in [0.01,0.003,0.001,0.0003,0.0001]:
    with open('../adversarial/%s/%s/CW_%s.pkl' %(args.dataset, args.model,e), 'rb') as f:
        adversarial = pickle.load(f)
    X = torch.tensor(np.concatenate((natural_test, adversarial['X'][:nval])))
    y = [0]*len(natural_test) + [1]*adversarial['X'][:nval].shape[0]
    detect_pred = detect(X,best['q1'],best['q2'],best['q3'],best['en1'],best['en2'])
    print(e,roc_auc_score(y,detect_pred))
