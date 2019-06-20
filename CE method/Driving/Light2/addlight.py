import time

from configs import bcolors
from utils import *

from scipy.misc import imsave

from driving_models import *
from utils import *
import numpy as np



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



def add_light(temp, gradients):
    import skimage
    temp = temp.reshape(temp.shape[0], -1)
    gradients = gradients.reshape(gradients.shape[0], -1)
    new_grads = np.ones_like(gradients)
    grad_mean = 500 * np.mean(gradients, axis=1)
    grad_mean = np.tile(grad_mean, temp.shape[1])
    grad_mean = grad_mean.reshape(temp.shape)
    temp = temp + 10 * new_grads * grad_mean
    temp = temp.reshape(temp.shape[0], 100, 100, 3)
    return temp


test, label = load_data()
x_test = test.copy()
y_test = label.copy()
img = deprocess_image(x_test[0])
imsave('orig.png', img)
temp = test.copy()
pert = 1 * np.random.normal(size=x_test.shape)
temp = add_light(temp, pert)
img = deprocess_image(temp[0])
imsave('pert.png', img)
x_test = temp.copy()
test = temp.copy()