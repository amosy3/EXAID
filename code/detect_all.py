from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse


def get_parsed_args():
    parser = argparse.ArgumentParser(description='This script creates explanations for net predictions based on shap. '
                                                 'Please choose dataset for natural samples, model, '
                                                 'and attack to explain')
    parser.add_argument('dataset', action='store', choices=['MNIST','CIFAR10'], type=str, help='dataset')
    parser.add_argument('model', action='store', choices=['resnet', 'vgg', 'googlenet'], type=str, help='model')
    parser.add_argument('adv_to_detect', action='store', type=str, help='which attack to use as test data')
    parser.add_argument('adv_to_train', action='store', type=str, choices=['None', 'FGSM1', 'FGSM03'],
                        help='let the detector see example for attack')
    args = parser.parse_args()
    return args


def load_data(args):

    with open('../explanations/%s/%s/natural.pkl' %(args.dataset,args.model), 'rb') as f:
        natural = pickle.load(f)

    with open('../explanations/%s/%s/%s.pkl' %(args.dataset,args.model,args.adv_to_detect), 'rb') as f:
        adv_to_detect = pickle.load(f)

    if args.adv_to_train == 'None':
        return natural, None, adv_to_detect

    with open('../explanations/%s/%s/%s.pkl' %(args.dataset,args.model,args.adv_to_train), 'rb') as f:
        adv_to_train = pickle.load(f)

    return natural, adv_to_train, adv_to_detect


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
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(filename+'.png')
    return roc_auc


def train_and_predict(model, X_train, y_train, X_test, y_test, filename='tmp'):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3000,
                        class_weight='auto', verbose=0)
    pred = model.predict(X_test)
    return plot_roc(y_test, pred, filename), history


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


args = get_parsed_args()
natural, adv_to_train, adv_to_detect = load_data(args.dataset,args.model)

scores = []
print('Train-%s Test-%s' %(args.adv_to_detect, args.adv_to_attack))
for n in range(10):
    print('======================== %d ========================' %n)

    all_exp = extract_all_exp(natural, adv_to_detect, adv_to_train, n)
    data = explanation2train_test(all_exp, adv_to_train)
    model = get_keras_model(data['X_train'])

    auc, history = train_and_predict(model, data['X_train'], data['y_train'], data['X_test'], data['y_test'],
                                               filename='Train-%s Test-%s'%(args.adv_to_detect,args.adv_to_attack))
    scores.append(np.round(auc,4))
    print('%d auc: %0.4f'%(n, auc))

print('Final auc= %0.4f' %np.mean(scores))
