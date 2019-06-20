import time

from configs import bcolors
from utils import *

from scipy.misc import imsave

from driving_models import *
from utils import *
import numpy as np

import matplotlib


def load_data():
    path = './testing/final_example.csv'
    temp = np.loadtxt(path, delimiter=',', dtype=np.str, skiprows=(1))
    names = list(temp[:, 0])
    test = []
    label = []
    for i in range(1):
        n = names[i]
        path = './testing/center/' + n + '.jpg'
        test.append(preprocess_image(path))
        label.append(float(temp[i, 1]))
    test = np.array(test)
    test = test.reshape(test.shape[0], 100, 100, 3)
    label = np.array(label)
    return test, label


def add_black(temp, gradients):
    rect_shape = (10, 10)
    for i in range(temp.shape[0]):
        orig = temp[i].reshape(1, 100, 100, 3)
        grad = gradients[i].reshape(1, 100, 100, 3)
        start_point = (
            random.randint(0, grad.shape[1] - rect_shape[0]), random.randint(0, grad.shape[2] - rect_shape[1]))
        new_grads = np.zeros_like(grad)
        patch = grad[:, start_point[0]:start_point[
            0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
                      start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
        orig = orig + 100 * new_grads
        temp[i] = orig.reshape(100, 100, 3)
    return temp


test, label = load_data()
x_test = test.copy()
y_test = label.copy()
img = deprocess_image(x_test[0])
imsave('orig.png', img)
temp = test.copy()
pert = 1 * np.random.normal(size=x_test.shape)
for i in range(5):
    temp = add_black(temp, pert)
img = deprocess_image(temp[0])
imsave('pert.png', img)
x_test = temp.copy()
test = temp.copy()