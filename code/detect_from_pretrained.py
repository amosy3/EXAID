from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def load_data(args,attack_to_detect):

    with open('../explanations/%s/%s/natural.pkl' %(args.dataset,args.model), 'rb') as f:
        natural = pickle.load(f)

    with open('../explanations/%s/%s/%s.pkl' %(args.dataset, args.model, attack_to_detect), 'rb') as f:
        adv_to_detect = pickle.load(f)

    return natural, None, adv_to_detect



def get_explained_image(data, n, explanation_type='good'):
    if explanation_type == 'good':  # data == natural_correctly classified
        ind = (data['label'] == n) & (data['net_pred'] == n)  # good explanation = label and pred sync.

    if explanation_type == 'weak':  # data == natural_correctly classified
        ind = (data['label'] != n) & (data['net_pred'] != n)  # weak = label and pred is not n -> bad explanation for n.

    if explanation_type == 'adversarial':  # data == adversarial
        ind = (data['label'] != n) & (data['net_pred'] == n)  # adversarial = the net explain why this non-8 is an 8=bad

    if explanation_type == 'wrong':  # data == natural_missclassifed
        ind = (data['label'] != n) & (data['net_pred'] == n)  # wrong = label and pred aren't sync.

    explained_image = np.concatenate((data['X'][[ind]], data['shap'][[ind]][:, n, :, :, :]), axis=1)
    return explained_image


def get_keras_model(X_train):
    from keras.models import Input, Model
    from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
    inputs = Input(shape=X_train.shape[1:])
    x = Conv2D(16, 3, strides=(1, 1), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.5)(x)

    x = Conv2D(32, 3, strides=(1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.5)(x)

    x = Conv2D(32, 3, strides=(1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.5)(x)

    x = Flatten()(x)
    x = Dense(30, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    pred = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=pred)
    # print(model.summary())
    return model


def plot_roc(y_test, pred, filename='tmp'):
    fpr, tpr, _ = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(filename.split('/')[-1])
    plt.legend(loc="lower right")
    plt.savefig(filename+'.png')
    return roc_auc


def train_and_predict(model, X_train, y_train, X_test, y_test, filename='tmp'):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', kauc])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3000,
                        class_weight='auto', verbose=0)
    pred = model.predict(X_test)
    save_history(history, filename)
    return plot_roc(y_test, pred, filename), history


def save_history(history,filename):
    # print(history.history)
    plt.figure()
    plt.plot(history.history['kauc'])
    plt.plot(history.history['val_kauc'])
    plt.title(filename)
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename + '_history.png')


def kauc(y_true, y_pred):
    kauc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return kauc


def explanation2train_test(all_exp, adv_to_train, channels_convection='WHC'):
    # 0-good explanation, 1-bad explanation
    y = np.array([0] * all_exp['good'].shape[0])
    good_exp_train, good_exp_test, _, _ = train_test_split(all_exp['good'], y, test_size=0.3)
    data = dict()

    if adv_to_train is None:
        # Do not enter adversarial to train! auc for each attack
        data['X_train'] = np.concatenate((all_exp['wrong'], all_exp['weak'], good_exp_train))
        data['y_train'] = np.array(
            [1] * (all_exp['wrong'].shape[0] + all_exp['weak'].shape[0]) + [0] * good_exp_train.shape[0])

        data['X_test'] = np.concatenate((all_exp['adv_detect'], good_exp_test))
        data['y_test'] = np.array([1] * all_exp['adv_detect'].shape[0] + [0] * good_exp_test.shape[0])

        if channels_convection == 'WHC':  # change CWH -> WHC - for keras
            data['X_train'] = np.transpose(data['X_train'], (0, 2, 3, 1))
            data['X_test'] = np.transpose(data['X_test'], (0, 2, 3, 1))

    else:
        data['X_train'] = np.concatenate((all_exp['wrong'], all_exp['weak'], all_exp['adv_train'], good_exp_train))
        data['y_train'] = np.array([1] * (all_exp['wrong'].shape[0] + all_exp['weak'].shape[0] +
                                          all_exp['adv_train'].shape[0]) +
                                   [0] * good_exp_train.shape[0])

        data['X_test'] = np.concatenate((all_exp['adv_detect'], good_exp_test))
        data['y_test'] = np.array([1] * all_exp['adv_detect'].shape[0] + [0] * good_exp_test.shape[0])

        if channels_convection == 'WHC':  # change CWH -> WHC - for keras
            data['X_train'] = np.transpose(data['X_train'], (0, 2, 3, 1))
            data['X_test'] = np.transpose(data['X_test'], (0, 2, 3, 1))

    return data


def extract_all_exp(natural, adv_to_detect, adv_to_train, n):
    all_exp = dict()
    all_exp['good'] = get_explained_image(natural, n, explanation_type='good')
    all_exp['weak'] = get_explained_image(natural, n, explanation_type='weak')
    all_exp['wrong'] = get_explained_image(natural, n, explanation_type='wrong')
    all_exp['adv_detect'] = get_explained_image(adv_to_detect, n, explanation_type='adversarial')
    if adv_to_train is not None:
        all_exp['adv_train'] = get_explained_image(adv_to_train, n, explanation_type='adversarial')

    return all_exp



def get_parsed_args():
    parser = argparse.ArgumentParser(description='This script detect adversarial examples with saved models')
    parser.add_argument('dataset', action='store',choices=['MNIST','CIFAR10', 'SVHN'], type=str, help='dataset')
    parser.add_argument('model', action='store',choices=['resnet', 'vgg', 'googlenet'], type=str, help='model')
    parser.add_argument('load_pretained_on', action='store',choices=['FGSM_0.1', 'None'], type=str, help='pretrain')
    args = parser.parse_args()
    return args


args = get_parsed_args()

# args = dict()
# args['dataset'] = 'SVHN'  # ['MNIST','CIFAR10', 'SVHN']
# args['model'] = 'resnet'
# # args['adv_to_detect'] = 'CW_0.0001' #CW:[0.01,0.003,0.001,0.0003,0.0001]. FGSM [0.3,0.1,0.03,0.01,0.003,0.001,0.0003]
# args['adv_to_train'] = 'None'  # ['None', 'FGSM_0.1', 'FGSM_0.03']

for attack in ['FGSM', 'PGD']:
    print(attack)
    for e in tqdm([0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003]):
        attack_to_detect = '%s_%s' % (attack, e)
        natural, adv_to_train, adv_to_detect = load_data(args, attack_to_detect)
        scores = []
        for n in (range(10)):
            all_exp = extract_all_exp(natural, adv_to_detect, adv_to_train, n)
            data = explanation2train_test(all_exp, adv_to_train)
            model = load_model(
                '../xai_models/%s/%s/%s_%s.h5' % (args.dataset, args.model, args.load_pretained_on, n),
                custom_objects={'kauc': kauc})
            pred = model.predict(data['X_test'])
            model_auc = roc_auc_score(data['y_test'], pred)

            scores.append(np.round(model_auc, 4))
        print('%s auc= %0.4f' % (attack_to_detect, np.mean(scores)))

print('CW')
for e in tqdm([0.01, 0.003, 0.001, 0.0003, 0.0001]):
    attack_to_detect = 'CW_%s' %e
    natural, adv_to_train, adv_to_detect = load_data(args, attack_to_detect)
    scores = []
    for n in (range(10)):
        all_exp = extract_all_exp(natural, adv_to_detect, adv_to_train, n)
        data = explanation2train_test(all_exp, adv_to_train)
        model = load_model('../xai_models/%s/%s/%s_%s.h5' % (args.dataset, args.model, args.load_pretained_on, n),
                           custom_objects={'kauc': kauc})
        pred = model.predict(data['X_test'])
        model_auc = roc_auc_score(data['y_test'], pred)

        scores.append(np.round(model_auc, 4))
    print('%s auc= %0.4f' % (attack_to_detect, np.mean(scores)))