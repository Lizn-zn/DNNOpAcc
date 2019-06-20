'''

'''
import numpy as np
import keras.backend as K
from collections import defaultdict
import matplotlib.pyplot as plt

import torch

def selectsample(model, x_test, y_test, delta, iterate):
    output = model.predict(x_test)
    max_output = np.array(torch.load('transfer_lsa.pt'))
    index = np.argsort(max_output)
    # devide into 3 strata
    index1 = index[0:int(x_test.shape[0] * 0.8)]
    index2 = index[int(x_test.shape[0] * 0.8):int(x_test.shape[0] * 0.9)]
    index3 = index[int(x_test.shape[0] * 0.9):]
    
    acc_list1 = []
    acc_list2 = []
    statra_index1 = []
    statra_index2 = []
    statra_index3 = []
    random_index = []
    for i in range(iterate):
        arr = np.random.permutation(index1.shape[0])
        temp_index = arr[0:int(delta * 0.4)]
        statra_index1 = np.append(statra_index1, index1[temp_index])
        statra_index1 = statra_index1.astype('int')
        label = y_test[statra_index1]
        orig_sample = x_test[statra_index1]
        orig_sample = orig_sample.reshape(-1, img_rows, img_cols, 1)
        pred = np.argmax(model.predict(orig_sample), axis=1)
        acc_1 = np.sum(pred == label) / orig_sample.shape[0]

        arr = np.random.permutation(index2.shape[0])
        temp_index = arr[0:int(delta * 0.4)]
        statra_index2 = np.append(statra_index2, index2[temp_index])
        statra_index2 = statra_index2.astype('int')
        label = y_test[statra_index2]
        orig_sample = x_test[statra_index2]
        orig_sample = orig_sample.reshape(-1, img_rows, img_cols, 1)
        pred = np.argmax(model.predict(orig_sample), axis=1)
        acc_2 = np.sum(pred == label) / orig_sample.shape[0]

        arr = np.random.permutation(index3.shape[0])
        temp_index = arr[0:int(delta * 0.2)]
        statra_index3 = np.append(statra_index3, index3[temp_index])
        statra_index3 = statra_index3.astype('int')
        label = y_test[statra_index3]
        orig_sample = x_test[statra_index3]
        orig_sample = orig_sample.reshape(-1, img_rows, img_cols, 1)
        pred = np.argmax(model.predict(orig_sample), axis=1)
        acc_3 = np.sum(pred == label) / orig_sample.shape[0]

        arr = np.random.permutation(x_test.shape[0])
        random_index = np.append(random_index, arr[0:delta])
        random_index = random_index.astype('int')

        acc1 = 0.1 * acc_1 + 0.1 * acc_2 + 0.8 * acc_3
        acc_list1.append(acc1)
        label = y_test[random_index]
        orig_sample = x_test[random_index]
        orig_sample = orig_sample.reshape(-1, img_rows, img_cols, 1)
        pred = np.argmax(model.predict(orig_sample), axis=1)
        acc2 = np.sum(pred == label) / orig_sample.shape[0]
        acc_list2.append(acc2)
        print("numuber of samples is {!s}, select acc is {!s}, random acc is {!s}".format(
            orig_sample.shape[0], acc1, acc2))
    return acc_list1, acc_list2


def experiments(delta, iterate):
    pred = np.argmax(model.predict(x_test), axis=1)
    true_acc = np.sum(pred == y_test) / x_test.shape[0]
    print("The final acc is {!s}".format(true_acc))

    acc_list1, acc_list2 = selectsample(
        model, x_test, y_test, delta, iterate)
    print("the select std is {!s}".format(
        np.mean(np.abs(acc_list1 - true_acc))))
    print("the random std is {!s}".format(
        np.mean(np.abs(acc_list2 - true_acc))))

    return np.array(acc_list1), np.array(acc_list2)


if __name__ == '__main__':
    # preprocess the data set
    from keras.datasets import mnist
    img_rows, img_cols = 16, 16
    import torch
    x_test, y_test = torch.load('test.pt')
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    from keras.models import load_model
    model = load_model('model/model_mnist.h5')


    for k in range(50):
        print("the {} exp".format(k))
        acc1, acc2 = experiments(delta=5, iterate=36)
        np.savetxt('data/select{}.csv'.format(k), acc1)
        np.savetxt('data/random{}.csv'.format(k), acc2)