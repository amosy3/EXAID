from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse


def get_parsed_args():
    parser = argparse.ArgumentParser(description='This script creates explanations for net predictions based on shap. '
                                                 'Please choose dataset for natural samples, model, '
                                                 'and attack to explain')
    parser.add_argument('dataset', action='store',choices=['MNIST','CIFAR10'], type=str, help='dataset')
    parser.add_argument('model', action='store',choices=['resnet', 'vgg', 'googlenet'], type=str, help='model')
    args = parser.parse_args()
    return args


def load_data(dataset,model):
    with open('../explanations/%s/%s/natural.pkl' %(dataset,model), 'rb') as f:
        natural = pickle.load(f)

    with open('../explanations/%s/%s/FGSM.pkl' %(dataset,model), 'rb') as f:
        adversarial_FGSM = pickle.load(f)

    with open('../explanations/%s/%s/PGD.pkl' %(dataset,model), 'rb') as f:
        adversarial_PGD = pickle.load(f)

    with open('../explanations/%s/%s/CW.pkl' %(dataset,model), 'rb') as f:
        adversarial_CW = pickle.load(f)

    return natural, adversarial_FGSM, adversarial_PGD, adversarial_CW


def get_explained_image(data, n, explanation_type):
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
    print(model.summary())
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
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1500, class_weight='auto')
    pred = model.predict(X_test)
    return plot_roc(y_test, pred, filename), history


def explanation2train_test(all_exp, mode='all_adv_in_test', channels_convection='WHC'):
    # 0-good explanation, 1-bad explanation
    y = np.array([0] * all_exp['good'].shape[0])
    good_exp_train, good_exp_test, _, _ = train_test_split(all_exp['good'], y, test_size=0.3)
    data = dict()
    if mode == 'all_adv_in_test':
        # Do not enter adversarial to train! auc for each attack
        data['X_train'] = np.concatenate((all_exp['wrong'], all_exp['weak'], good_exp_train))
        data['y_train'] = np.array(
            [1] * (all_exp['wrong'].shape[0] + all_exp['weak'].shape[0]) + [0] * good_exp_train.shape[0])

        data['X_test_FGSM'] = np.concatenate((all_exp['adv_FGSM'], good_exp_test))
        data['y_test_FGSM'] = np.array([1] * all_exp['adv_FGSM'].shape[0] + [0] * good_exp_test.shape[0])

        data['X_test_PGD'] = np.concatenate((all_exp['adv_PGD'], good_exp_test))
        data['y_test_PGD'] = np.array([1] * all_exp['adv_PGD'].shape[0] + [0] * good_exp_test.shape[0])

        data['X_test_CW'] = np.concatenate((all_exp['adv_CW'], good_exp_test))
        data['y_test_CW'] = np.array([1] * all_exp['adv_CW'].shape[0] + [0] * good_exp_test.shape[0])

        if channels_convection == 'WHC':  # change CWH -> WHC - for keras
            data['X_train'] = np.transpose(data['X_train'], (0, 2, 3, 1))
            data['X_test_FGSM'] = np.transpose(data['X_test_FGSM'], (0, 2, 3, 1))
            data['X_test_PGD'] = np.transpose(data['X_test_PGD'], (0, 2, 3, 1))
            data['X_test_CW'] = np.transpose(data['X_test_CW'], (0, 2, 3, 1))

    if mode == 'leave_FGSM_out':
        # Do not enter adversarial to train! auc for each attack
        data['X_train'] = np.concatenate((all_exp['wrong'], all_exp['weak'], all_exp['adv_PGD'],
                                          all_exp['adv_CW'], good_exp_train))
        data['y_train'] = np.array([1] * (all_exp['wrong'].shape[0] + all_exp['weak'].shape[0] +
                                          all_exp['adv_PGD'].shape[0] + all_exp['adv_CW'].shape[0]) +
                                   [0] * good_exp_train.shape[0])

        data['X_test_FGSM'] = np.concatenate((all_exp['adv_FGSM'], good_exp_test))
        data['y_test_FGSM'] = np.array([1] * all_exp['adv_FGSM'].shape[0] + [0] * good_exp_test.shape[0])

        if channels_convection == 'WHC':  # change CWH -> WHC - for keras
            data['X_train'] = np.transpose(data['X_train'], (0, 2, 3, 1))
            data['X_test_FGSM'] = np.transpose(data['X_test_FGSM'], (0, 2, 3, 1))

    if mode == 'leave_PGD_out':
        # Do not enter adversarial to train! auc for each attack
        data['X_train'] = np.concatenate((all_exp['wrong'], all_exp['weak'], all_exp['adv_FGSM'],
                                          all_exp['adv_CW'], good_exp_train))
        data['y_train'] = np.array([1] * (all_exp['wrong'].shape[0] + all_exp['weak'].shape[0] +
                                          all_exp['adv_FGSM'].shape[0] + all_exp['adv_CW'].shape[0]) +
                                   [0] * good_exp_train.shape[0])

        data['X_test_PGD'] = np.concatenate((all_exp['adv_PGD'], good_exp_test))
        data['y_test_PGD'] = np.array([1] * all_exp['adv_PGD'].shape[0] + [0] * good_exp_test.shape[0])

        if channels_convection == 'WHC':  # change CWH -> WHC - for keras
            data['X_train'] = np.transpose(data['X_train'], (0, 2, 3, 1))
            data['X_test_PGD'] = np.transpose(data['X_test_PGD'], (0, 2, 3, 1))

    if mode == 'leave_CW_out':
        # Do not enter adversarial to train! auc for each attack
        data['X_train'] = np.concatenate((all_exp['wrong'], all_exp['weak'], all_exp['adv_FGSM'],
                                          all_exp['adv_PGD'], good_exp_train))
        data['y_train'] = np.array([1] * (all_exp['wrong'].shape[0] + all_exp['weak'].shape[0] +
                                          all_exp['adv_FGSM'].shape[0] + all_exp['adv_PGD'].shape[0]) +
                                   [0] * good_exp_train.shape[0])

        data['X_test_CW'] = np.concatenate((all_exp['adv_CW'], good_exp_test))
        data['y_test_CW'] = np.array([1] * all_exp['adv_CW'].shape[0] + [0] * good_exp_test.shape[0])

        if channels_convection == 'WHC':  # change CWH -> WHC - for keras
            data['X_train'] = np.transpose(data['X_train'], (0, 2, 3, 1))
            data['X_test_CW'] = np.transpose(data['X_test_CW'], (0, 2, 3, 1))

    if mode == 'split_all':
        # Do not enter adversarial to train! auc for each attack

        y = np.array([0] * all_exp['adv_FGSM'].shape[0])
        FGSM_train, FGSM_test, _, _ = train_test_split(all_exp['adv_FGSM'], y, test_size=0.3)

        y = np.array([0] * all_exp['adv_PGD'].shape[0])
        PGD_train, PGD_test, _, _ = train_test_split(all_exp['adv_PGD'], y, test_size=0.3)

        y = np.array([0] * all_exp['adv_CW'].shape[0])
        CW_train, CW_test, _, _ = train_test_split(all_exp['adv_CW'], y, test_size=0.3)

        data['X_train'] = np.concatenate((all_exp['wrong'], all_exp['weak'],
                                          FGSM_train, PGD_train, CW_train, good_exp_train))
        data['y_train'] = np.array([1] * (all_exp['wrong'].shape[0] + all_exp['weak'].shape[0] +
                                          FGSM_train.shape[0] + PGD_train.shape[0] + CW_train.shape[0]) +
                                   [0] * good_exp_train.shape[0])

        data['X_test'] = np.concatenate((FGSM_test, PGD_test, CW_test, good_exp_test))
        data['y_test'] = np.array([1] * (FGSM_test.shape[0] + PGD_test.shape[0] + CW_test.shape[0]) +
                                  [0] * good_exp_test.shape[0])

        if channels_convection == 'WHC':  # change CWH -> WHC - for keras
            data['X_train'] = np.transpose(data['X_train'], (0, 2, 3, 1))
            data['X_test'] = np.transpose(data['X_test'], (0, 2, 3, 1))
    return data


args = get_parsed_args()
natural, adversarial_FGSM, adversarial_PGD, adversarial_CW = load_data(args.dataset,args.model)

n = 4

all_exp = dict()
all_exp['good'] = get_explained_image(natural, n, expanation_type='good')
all_exp['weak'] = get_explained_image(natural, n, expanation_type='weak')
all_exp['adv_FGSM'] = get_explained_image(adversarial_FGSM, n, expanation_type='adversarial')
all_exp['adv_PGD'] = get_explained_image(adversarial_PGD, n, expanation_type='adversarial')
all_exp['adv_CW'] = get_explained_image(adversarial_CW, n, expanation_type='adversarial')
all_exp['wrong'] = get_explained_image(natural, n, expanation_type='wrong')


data = explanation2train_test(all_exp, mode='all_adv_in_test')
model = get_keras_model(data['X_train'])
auc_FGSM, history_FGSM = train_and_predict(model, data['X_train'], data['y_train'],
                                           data['X_test_FGSM'], data['y_test_FGSM'], filename='FGSM')
print('FGSM auc: %0.2f' %auc_FGSM)
auc_PGD, history_PGD = train_and_predict(model, data['X_train'], data['y_train'],
                                         data['X_test_PGD'], data['y_test_PGD'], filename='PGD')
print('PGD auc: %0.2f' %auc_PGD)
auc_CW, history_CW = train_and_predict(model, data['X_train'], data['y_train'],
                                       data['X_test_CW'], data['y_test_CW'], filename='CW')
print('CW auc: %0.2f' %auc_CW)

for attack in ['FGSM','PGD','CW']:
    data = explanation2train_test(all_exp, mode='leave_%s_out' %attack)
    model = get_keras_model(data['X_train'])
    attack_auc, attack_history = train_and_predict(model, data['X_train'], data['y_train'],
                                               data['X_test_%s' %attack], data['y_test_%s' %attack],
                                                   filename='leave_%s_out' %attack)
    print('$s auc: %0.2f' %(attack, attack_auc))

data = explanation2train_test(all_exp, mode='split_all')
model = get_keras_model(data['X_train'])
split_all_auc, split_all_history = train_and_predict(model, data['X_train'], data['y_train'],
                                           data['X_test'], data['y_test'],
                                               filename='split_all')
print('split all auc: %0.2f' %(split_all_auc))