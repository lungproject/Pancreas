from __future__ import print_function

import datetime
import keras
import numpy as np
from keras.callbacks import CSVLogger
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
# from keras.applications import ResNet50

# from keras.applications.inception_v3 import InceptionV3
# from keras.models import  Sequential
from keras.layers import Input,Dense, Dropout, Flatten, Activation,AveragePooling2D, GlobalAveragePooling2D,BatchNormalization,GlobalMaxPooling2D,UpSampling2D
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras import regularizers
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.cross_validation import StratifiedKFold
from keras.models import load_model
from keras import backend as K
import tensorflow as tf

from Loaddata import load_data1,load_data2,load_data3,load_data4

import matplotlib.pyplot as plt  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#from Loaddata import load_data

from sklearn.metrics import roc_auc_score
from scipy.io import loadmat



# from sklearn.cross_validation import train_validate_split
class LossHistory(keras.callbacks.Callback):  
    def on_train_begin(self, logs={}):  
        self.losses = {'batch':[], 'epoch':[]}  
        self.accuracy = {'batch':[], 'epoch':[]}  
        self.val_loss = {'batch':[], 'epoch':[]}  
        self.val_acc = {'batch':[], 'epoch':[]}  

    def on_batch_end(self, batch, logs={}):  
        self.losses['batch'].append(logs.get('loss'))  
        self.accuracy['batch'].append(logs.get('acc'))  
        self.val_loss['batch'].append(logs.get('val_loss'))  
        self.val_acc['batch'].append(logs.get('val_acc'))  

    def on_epoch_end(self, batch, logs={}):  
        self.losses['epoch'].append(logs.get('loss'))  
        self.accuracy['epoch'].append(logs.get('acc'))  
        self.val_loss['epoch'].append(logs.get('val_loss'))  
        self.val_acc['epoch'].append(logs.get('val_acc'))  

    def loss_plot(self, loss_type):  
        iters = range(len(self.losses[loss_type]))  
        plt.figure()  
        # acc  
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')  
        # loss  
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')  
        if loss_type == 'epoch':  
            # val_acc  
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')  
            # val_loss  
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')  
        plt.grid(True)  
        plt.xlabel(loss_type)  
        plt.ylabel('acc-loss')  
        plt.legend(loc="upper right")  
        plt.show()
# def binary_focal_loss(gamma=2, alpha=0.25):
#     """
#     Binary form of focal loss.
#     适用于二分类问题的focal loss
    
#     focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
#         where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
#     References:
#         https://arxiv.org/pdf/1708.02002.pdf
#     Usage:
#      model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """
#     alpha = tf.constant(alpha, dtype=tf.float32)
#     gamma = tf.constant(gamma, dtype=tf.float32)

#     def binary_focal_loss_fixed(y_true, y_pred):
#         """
#         y_true shape need be (None,1)
#         y_pred need be compute after sigmoid
#         """
#         y_true = tf.cast(y_true, tf.float32)
#         alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
#         p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
#         focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
#         return K.mean(focal_loss)
#     return binary_focal_loss_fixed
def binary_focal_loss(gamma=2., alpha=.25):
    
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        pt_1 = K.clip(pt_1, 1e-3, .999)
        pt_0 = K.clip(pt_0, 1e-3, .999)
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
# def auc(y_true, y_pred):
#     auc_value, auc_op = tf.metrics.auc(y_true, y_pred)#[1]
#     K.get_session().run(tf.global_variables_initializer())
#     K.get_session().run(tf.local_variables_initializer())
#     K.get_session().run(auc_op)
#     auc = K.get_session().run(auc_value)

#     return auc
#FUSE CROSS VALIDATION
now = datetime.datetime.now
print("validate0")

# batch_size = 32
# num_classes = 2




#print(y_validate.shape)

# input image dimensions
img_rows, img_cols = 64, 64
# number of convolutional filters to use
# size of pooling area for max pooling
pool_size = 2
# convolution kernel size
kernel_size = 3
history = LossHistory() 
input_shape = (img_rows, img_cols, 1)

num_classes=2
epochs = 5000
batch_size=128
numfilter=4


#ResNet Block
def conv_block(inputs,num_filters=4, kernel_size=3,strides=1, activation='relu',padding='same', dilation_rate=(1, 1)):
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding=padding,dilation_rate=dilation_rate,
            kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(1e-3))(inputs)
    x = BatchNormalization()(x)
    if(activation):
        x = Activation('relu')(x)
        # x = PReLU(alpha_initializer='zeros', weights=None)(x)
    # x = Dropout(0.2)(x)   
    # x = BatchNormalization()(x)    
    return x

def resnet_block(inputs,num_filters=4,strides=1,activation='relu',kernel_size=3,padding='same', dilation_rate=(1, 1)):
    
    a = conv_block(inputs,strides=strides,num_filters=num_filters,kernel_size=3,padding=padding, dilation_rate=dilation_rate)
    b = conv_block(inputs= a,activation=None,num_filters=num_filters,kernel_size=3,padding=padding, dilation_rate=dilation_rate)
    
    x = Conv2D(num_filters,kernel_size=3,strides=strides,padding=padding,dilation_rate=dilation_rate,
                       kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(1e-3))(inputs)
    x = keras.layers.add([x,b])
    x = BatchNormalization()(x)   
    if(activation):
        x = Activation('relu')(x)
        # x = PReLU(alpha_initializer='zeros', weights=None)(x)
    # x = BatchNormalization()(x)    
    # x = Dropout(0.2)(x)      
    return x



  


def resnet_v1(input_shape,num_filters=8):
   
    
    inputs = Input(shape=input_shape)
 

    x = conv_block(inputs,num_filters=num_filters,kernel_size=3,strides=2)  
        
    x = resnet_block(x,num_filters=num_filters)    
    x = resnet_block(x,num_filters=num_filters,strides=1,dilation_rate=1)
    # x = conv_block(inputs = x,num_filters=num_filters,strides=1)

    x = resnet_block(x,num_filters=num_filters,strides=2)
    x = resnet_block(inputs = x,num_filters=num_filters,strides=1,dilation_rate=1)
    # x = conv_block(inputs = x,num_filters=num_filters*2,strides=1,dilation_rate=3)

    x = resnet_block(x,num_filters=num_filters*2,strides=2)
    x = resnet_block(inputs = x,num_filters=num_filters*2,strides=1,dilation_rate=1)
    # x = conv_block(inputs = x,num_filters=num_filters*2,strides=1,dilation_rate=3)

    x = resnet_block(inputs = x,num_filters=num_filters*4,strides=1)
    x = resnet_block(inputs = x,num_filters=num_filters*8,strides=1)
    # x = MaxPooling2D(pool_size=2)(x)
    # x = resnet_block(inputs = x,num_filters=num_filters*4,strides=1)
    y = GlobalAveragePooling2D()(x)
    # y = Flatten()(x)
    y = Dropout(0.5)(y)
    # 
    # out:2*2*64
    # 
    

    # out:1024
    outputs = Dense(num_classes,activation='softmax',
                    kernel_initializer='he_normal')(y)
    
    #初始化模型
    #之前的操作只是将多个神经网络层进行了相连，通过下面这一句的初始化操作，才算真正完成了一个模型的结构初始化
    model = Model(inputs= inputs,outputs=outputs)
    return model


# def resnet_v2(input_shape,num_filters=16):
       
    
#     inputs = Input(shape=input_shape)


#     input32= conv_block(inputs,num_filters=num_filters,kernel_size=1,strides=2)
#     input16= conv_block(inputs,num_filters=num_filters,kernel_size=1,strides=4)  
#     # input8= conv_block(inputs,num_filters=num_filters*4,kernel_size=1,strides=8)  

#     x = conv_block(inputs,num_filters=num_filters,kernel_size=3,strides=2) 
#     # x = keras.layers.concatenate([inputpet32,inputct32,inputfuse32,x], axis = 3)  
#     x = resnet_block(x,num_filters=num_filters)    
#     x = resnet_block(x,num_filters=num_filters,strides=1,dilation_rate=1)
#     # x = conv_block(inputs = x,num_filters=num_filters,strides=1)

#     x = keras.layers.concatenate([input32,x], axis = 3)  
#     x = resnet_block(x,num_filters=num_filters,strides=2)
#     x = resnet_block(inputs = x,num_filters=num_filters,strides=1,dilation_rate=1)
#     # x = conv_block(inputs = x,num_filters=num_filters*2,strides=1,dilation_rate=3)

#     x = keras.layers.concatenate([input16,x], axis = 3)  
#     x = resnet_block(x,num_filters=num_filters*2,strides=2)
#     x = resnet_block(inputs = x,num_filters=num_filters*2,strides=1,dilation_rate=1)
#     # x = conv_block(inputs = x,num_filters=num_filters*2,strides=1,dilation_rate=3)

#     # x = keras.layers.concatenate([input8,x], axis = 3)  
#     x = resnet_block(inputs = x,num_filters=num_filters*16,strides=1)
#     x = resnet_block(inputs = x,num_filters=num_filters*32,strides=1)
#     # x = resnet_block(inputs = x,num_filters=num_filters*32,strides=1)
#     # x = MaxPooling2D(pool_size=2)(x)
#     # x = resnet_block(inputs = x,num_filters=num_filters*4,strides=1)
#     y = GlobalAveragePooling2D()(x)
#     # y = Flatten()(x)
#     y = Dropout(0.3)(y)
#     # 
#     # out:2*2*64
#     # 
    

#     # out:1024
#     outputs = Dense(num_classes,activation='softmax',
#                     kernel_initializer='he_normal')(y)
    
#     #初始化模型
#     #之前的操作只是将多个神经网络层进行了相连，通过下面这一句的初始化操作，才算真正完成了一个模型的结构初始化
#     model = Model(inputs=inputs,outputs=outputs)
#     return model    







x_train ,y_train ,x_test, y_test = load_data4()#load_data(count)


x_train = np.expand_dims(x_train, axis=3)
x_test= np.expand_dims(x_test, axis=3)

s = np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]
print(y_train.shape)
print(x_train.shape)
print(np.mean(y_test))
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)    

for i in range(4):
    

    model = resnet_v1((img_rows, img_cols,1),num_filters=16)
    # model.summary()
    

    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    # model.compile(loss=[binary_focal_loss(alpha=.5, gamma=2)], optimizer=adam, metrics=['accuracy'])

    t = now()



    cvscores = []
    pretrain = []
    pretest = []
    truetrain=[]
    truetest=[]
    count =0


   

    filepath="4res1cnnweightspatient2-improvement-{epoch:02d}-{acc:.2f}-{val_acc:.2f}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
    mode='min')
    # csv_logger = CSVLogger('training.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=1e-20)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=50)     
    callbacks_list = [checkpoint,earlyStopping,history]
        

    datagen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            width_shift_range=0.15,
            height_shift_range=0.15,           
            shear_range=0.1,
            rotation_range=90,
            zoom_range =0.1)#,
            # fill_mode='constant', 
            # cval=0.0)

    datagen.fit(x_train)


    history_callback=model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train)/batch_size,
            epochs=epochs,validation_data=(x_test, y_test),
            verbose=2,callbacks=callbacks_list)



model.save("weightspatient2-improvement"+str(count)+".hdf5")
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# history.loss_plot('epoch') 
cvscores.append(scores[1] * 100)

print('Training time: %s' % (now() - t))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

loss_history = history_callback.history["loss"]

np_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", np_loss_history, delimiter=",")

val_loss_history = history_callback.history["val_loss"]

np_val_loss_history = np.array(val_loss_history)
np.savetxt("val_loss_history.txt", np_val_loss_history, delimiter=",")

acc_history = history_callback.history["acc"]

np_acc_history = np.array(acc_history)
np.savetxt("acc_history.txt", np_acc_history, delimiter=",")

val_acc_history = history_callback.history["val_acc"]

np_val_acc_history = np.array(val_acc_history)
np.savetxt("val_acc_history.txt", np_val_acc_history, delimiter=",")