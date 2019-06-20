'''
LeNet-5
'''
# usage: python MNISTModel.py - train the model

from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
from keras import regularizers
import keras.backend as K

import numpy as np


def fgsm(model, x_test, eps=0.1):
    label = np.argmax(model.predict(x_test), axis=1)
    input_tensor = model.input
    from keras.utils.np_utils import to_categorical
    target = to_categorical(label, num_classes=model.output_shape[1])

    output = model.layers[-1].output
    fun = K.function([input_tensor], [output])
    loss = K.categorical_crossentropy(target, output, from_logits=False)
    grads = K.gradients(loss, input_tensor)[0]
    iterate = K.function([input_tensor], [loss, grads])

    loss, grads = iterate([x_test])
    normalized_grad = np.sign(grads)
    adv = np.clip(x_test + eps * normalized_grad, 0, 1)
    return adv


def train_model(file_name='Model'):
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)

    batch_size = 256
    nb_epoch = 5

    # input image dimensions
    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    index1 = np.where(y_train == 3)
    index2 = np.where(y_train == 9)
    # index3 = np.where(y_train == 7)
    index1 = index1[0][0: int(index1[0].shape[0])]
    y_train[index1] = 9
    index2 = index2[0][0: int(index2[0].shape[0])]
    y_train[index2] = 3
    # index3 = index3[0][0: int(index3[0].shape[0])]
    # y_train[index3] = 1


    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    input_tensor = Input(shape=input_shape)

    x = Convolution2D(8, kernel_size, activation='relu',
                      padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    x = Convolution2D(12, kernel_size, activation='relu',
                      padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1',)(x)
    x = Dense(84, activation='relu', name='fc2',)(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    """
    def Temp_Loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, K.softmax(y_pred / 100))

    compiling
    model.compile(loss=Temp_Loss,
                  optimizer='adadelta', metrics=['accuracy'])
    """
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta', metrics=['accuracy'])

    # define callback function
    import keras

    class MyCallback(keras.callbacks.Callback):

        def on_train_begin(self, logs={}):
            self.scores = []
            return

        def on_epoch_end(self, batch, logs={}):
            x_val = self.validation_data[0]
            y_val = self.validation_data[1]
            score = []
            pred = np.argmax(model.predict(x_val), axis=1)
            y_label = np.argmax(y_val, axis=1)
            for i in range(10):
                index = np.where(y_label == i)
                score.append(np.sum(pred[index] == i) / len(index[0]))
            self.scores.append(score)
            return

    cb = MyCallback()

    # trainig
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=batch_size, epochs=nb_epoch, callbacks=[cb])

    model.save('model/' + file_name + '.h5')

    score = model.evaluate(x_test, y_test, verbose=0)
    print('model is saved in {0}'.format('model/' + file_name + '.h5'))
    print('\n')
    print('Overall Test score:', score[0])
    print('Overall Test accuracy:', score[1])

    return model


if __name__ == '__main__':
    model = train_model(file_name='mutant')
