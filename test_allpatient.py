from __future__ import print_function

import datetime
import keras
import numpy as np
import tensorflow as tf
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.models import  Sequential
from keras.layers import Input,Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D,BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping,ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.cross_validation import StratifiedKFold

from keras import backend as K

from Loaddata_test import load_sample
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math



def binary_focal_loss(gamma=2., alpha=.25):
    
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        pt_1 = K.clip(pt_1, 1e-3, .999)
        pt_0 = K.clip(pt_0, 1e-3, .999)
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

model = load_model('./model/Pancreas.hdf5',custom_objects={'binary_focal_loss': binary_focal_loss, 'focal_loss_fixed': binary_focal_loss()})

x_test  = load_sample()
x_test = np.expand_dims(x_test, axis=3)

predicttest = model.predict(x_test, verbose=1)#ADCdata1,,xfuse_train
np.savetxt("./Results/predicttest.txt",predicttest )




