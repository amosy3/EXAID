import sys
import matplotlib.mlab as mlab
import scipy.integrate as integrate
from PIL import Image
from scipy import fft
from scipy import misc
from skimage import transform
import shutil
import requests
import tempfile
import os
import math
import pickle
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import roc_auc_score
import torch, torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from hyperopt import hp, tpe, fmin


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

    with open('../../adversarial/%s/%s/FGSM.pkl' %(dataset,model), 'rb') as f:
        adversarial_FGSM = pickle.load(f)

    with open('../../adversarial/%s/%s/PGD.pkl' %(dataset,model), 'rb') as f:
        adversarial_PGD = pickle.load(f)

    with open('../../adversarial/%s/%s/CW.pkl' %(dataset,model), 'rb') as f:
        adversarial_CW = pickle.load(f)

    return adversarial_FGSM, adversarial_PGD, adversarial_CW


def get_pretrain_model(model_name):
    if model_name == 'vgg':
        model_path = '../../models/vgg_acc_85.pkl'

    if model_name == 'googlenet':
        model_path = '../../models/googlenet_acc_84.pkl'

    if model_name == 'resnet':
        model_path = '../../models/resnetxt_acc_87.pkl'

    with open(model_path, 'rb') as f:
        net = pickle.load(f)
    return net.module

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

model = get_pretrain_model('resnet')
# adversarial_FGSM, adversarial_PGD, adversarial_CW = load_data('CIFAR10','resnet')
testloader = get_test_loader('CIFAR10')


def numpy2torch(x):
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    return torch.tensor(x).float().to('cuda')


def detect(data, q1=128, q2=64, q3=43, en1=4, en2=5):
    detection_pred = []
    for image in data:
        raw_pred = torch.argmax(model(numpy2torch(image)), dim=1)
        ori_entropy = oneDEntropy(image)
        #         print(ori_entropy)
        #         image = np.transpose(image,(1,2,0))
        if ori_entropy < en1:
            ppimage = scalarQuantization(image, q1)

        elif ori_entropy < en2:
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

detect_natural = []
i=0
for data,label in tqdm(testloader):
    detect_natural += detect(data)

print('dataset contain %d natural samples. %d falsely detected as adversarial'
      %(len(detect_natural), sum(detect_natural)))

with open('../../adversarial/%s/%s/FGSM_0.1.pkl' % ('CIFAR10', 'resnet'), 'rb') as f:
    adversarial_FGSM = pickle.load(f)


def objective(params):
    print(params)
    detect_FGSM = detect(torch.tensor(adversarial_FGSM['X'][:100]),
                         params['q1'], params['q2'], params['q3'], params['en1'], params['en2'])
    y = [0] * len(detect_natural) + [1] * len(detect_FGSM)

    return 1 - roc_auc_score(y, detect_natural + detect_FGSM)


space = {
    'q1': hp.choice('q1', range(100)),
    'q2': hp.choice('q2', range(100)),
    'q3': hp.choice('q3', range(100)),
    'en1': hp.choice('en1', range(10)),
    'en2': hp.choice('en2', range(10))
}

best = fmin(fn = objective, space =space, algo=tpe.suggest, max_evals = 100)

print('FGSM')
for e in [0.3,0.1,0.03,0.01,0.003,0.001,0.0003]:
    with open('../../adversarial/%s/%s/FGSM_%s.pkl' %('CIFAR10','resnet',e), 'rb') as f:
        adversarial_FGSM = pickle.load(f)
    detect_FGSM = detect(torch.tensor(adversarial_FGSM['X'][:2000]),best['q1'],best['q2'],best['q3'],best['en1'],best['en2'])
    y = [0]*len(detect_natural) + [1]*len(detect_FGSM)
    print(e,roc_auc_score(y,detect_natural+detect_FGSM))

print('PGD')
for e in [0.3,0.1,0.03,0.01,0.003,0.001,0.0003]:
    with open('../../adversarial/%s/%s/PGD_%s.pkl' %('CIFAR10','resnet',e), 'rb') as f:
        adversarial = pickle.load(f)
    sample_detect = detect(torch.tensor(adversarial['X'][:2000]),best['q1'],best['q2'],best['q3'],best['en1'],best['en2'])
    y = [0]*len(detect_natural) + [1]*len(sample_detect)
    print(e,roc_auc_score(y,detect_natural+sample_detect))