# imports

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pickle
from tqdm import tqdm
import glob
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt


# functions
def data_and_pred_channel(data,channel_vec):
    return np.array([data[i,[0,c+1],:,:] for i,c in enumerate(channel_vec)])

    
def choose_one_channel(data,channel_vec,keep_dim=True):
    x = np.array([data[i,c+1,:,:] for i,c in enumerate(channel_vec)])
    if keep_dim:
        return np.expand_dims(np.array(x), axis=1)
    return x


def normalize_single_cannel(data):
    div = np.sum(data,axis=2)
    div = np.sum(div,axis=2)
    return np.transpose(np.transpose(data,(2,1,0,3))/div,(2,1,0,3))


def flat_all_samples(np_data):
    return np_data.reshape([np_data.shape[0],-1])

def calc_out_layer_size(width, filter_size, stride=1, padding=0):
    return int((width-filter_size+2*padding)/stride + 1)

    
from sklearn.metrics import roc_curve, auc
def plot_roc(y_test, pred):
    fpr, tpr, _ = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    print('auc_score:', roc_auc)



# Load Data

with open('../data/final_db/natural_classified_correctly.pkl', 'rb') as f:
    natural = pickle.load(f)

with open('../data/final_db/natural_misclassified.pkl', 'rb') as f:
    misclassified = pickle.load(f)
    
with open('../data/final_db/adversarial_cv20:36:01.418627.pkl', 'rb') as f:
    adversarial = pickle.load(f)




# creat dataset

n=4

correct_on_my_class = natural['X'][[natural['label']==n]] #get the samples of n
correct_on_my_class_exp = natural['shap'][[natural['label']==n]] #get the samples of n
good_exp = np.concatenate((correct_on_my_class, correct_on_my_class_exp[:,n,:,:,:]), axis=1)
print(good_exp.shape)

correct_on_other_class = natural['X'][[natural['label']!=n]] #get non-n num data
correct_on_other_class_exp = natural['shap'][[natural['label']!=n]] #get the samples of n
weak_exp = np.concatenate((correct_on_other_class, correct_on_other_class_exp[:,n,:,:,:]), axis=1)
print(weak_exp.shape)

adv_target_to_my_class = adversarial['X'][[adversarial['net_pred']==n]]
adv_target_to_my_class_exp = adversarial['shap'][[adversarial['net_pred']==n]]
adv_exp = np.concatenate((adv_target_to_my_class, adv_target_to_my_class_exp[:,n,:,:,:]), axis=1)
print(adv_exp.shape)

misclassified_my_class = misclassified['X'][[misclassified['net_pred']==n]] #get non-n num data
misclassified_my_class_exp = misclassified['shap'][[misclassified['net_pred']==n]] #get the samples of n
wrong_exp = np.concatenate((misclassified_my_class, misclassified_my_class_exp[:,n,:,:,:]), axis=1)
print(wrong_exp.shape)



y = np.array([0]*good_exp.shape[0])
good_exp_train, good_exp_test, good_exp_y_train, good_exp_y_test = train_test_split(good_exp, y, test_size=0.3)

X_train = np.concatenate((wrong_exp, weak_exp, good_exp_train))
y_train = np.array([1]*(wrong_exp.shape[0]+weak_exp.shape[0])+[0]*good_exp_train.shape[0])

X_test = np.concatenate((adv_exp,good_exp_test))
y_test =np.array([1]*adv_exp.shape[0]+[0]*good_exp_test.shape[0])





from keras.models import Input, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# change CWH -> WHC
X_train = np.transpose(X_train, (0, 2, 3, 1))
X_test = np.transpose(X_test, (0, 2, 3, 1))


inputs = Input(shape=X_train.shape[1:])
x = Conv2D(16,3,strides=(1, 1), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.5)(x)

x = Conv2D(32,3,strides=(1, 1), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.5)(x)

x = Conv2D(32,3,strides=(1, 1), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.5)(x)

x = Flatten()(x)
x = Dense(30, activation='relu')(x)
x = Dropout(rate=0.5)(x)
pred = Dense(1, activation='sigmoid')(x)



model = Model(inputs=inputs, outputs=pred)
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1500, class_weight='auto')

pred = model.predict(X_test)
plot_roc(y_test,pred)







