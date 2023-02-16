import numpy as np
import os
import h5py

from time import time
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout, Input, TimeDistributed, Activation, concatenate, Reshape, Lambda, multiply, subtract, Conv2D, AveragePooling2D, Flatten
from tensorflow.keras.applications import VGG16

from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import  preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import GRU
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, roc_auc_score


import segmentation_models as sm
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16


""" define preprocess method condisering back_bone"""
preprocess_input = sm.get_preprocessing('vgg16')
""" model parameters"""
n_classes = 1 if len(Defined_CLASSES)== 1 else (len(Defined_CLASSES))
activation = 'sigmoid' if n_classes == 1 else 'softmax'

metrics = [tf.keras.metrics.AUC()]

def CNN_MODEL(input_image_dim):
    
    vgg16 =  VGG16(include_top = False, weights='imagenet')
    for layers in vgg16.layers:
        layers.trainable = False
#    vgg16.summary()
    input_x1 = Input(shape=input_image_dim, name = 'image_input1')
    
    x1 = Model(vgg16.input, vgg16.get_layer('block5_pool').output)(input_x1)
    x1 = Conv2D(4096, (1,1), padding= 'same', activation = 'relu')(x1)
    x1 = AveragePooling2D(pool_size=(7,7))(x1)
    x1 = Reshape((1,4096))(x1)
    
    input_x2 = Input(shape = input_image_dim, name = 'image_input2')
    x2 = Model(vgg16.input, vgg16.get_layer('block5_pool').output)(input_x2)
    x2 = Conv2D(4096, (1,1), padding= 'same', activation = 'relu')(x2)
    x2 = AveragePooling2D(pool_size=(7,7))(x2)
    x2 = Reshape((1,4096))(x2)
    
    x3 = subtract([x1,x2])
    
    model = Model([input_x1, input_x2],x3, name= 'mutliplication')
    
    return model

def create_model(metrics,input_image_dim, latent_dim, input_dim):
    ###########################################################################################################################
    # cc_views 
    ###########################################################################################################################
    input_cc_x1 = Input(shape=input_image_dim, name = 'CC_input1')
    input_cc_x2 = Input(shape = input_image_dim, name = 'CC_input2')
    
    input_cc_x3 = Input(shape=input_image_dim, name = 'CC_input3')
    input_cc_x4 = Input(shape = input_image_dim, name = 'CC_input4')
    
    input_cc_x5 = Input(shape=input_image_dim, name = 'CC_input5')
    input_cc_x6 = Input(shape = input_image_dim, name = 'CC_input6')
    
    input_cc_x7 = Input(shape=input_image_dim, name = 'CC_input7')
    input_cc_x8 = Input(shape = input_image_dim, name = 'CC_input8')
    
    cnn_unit_cc = CNN_MODEL(input_image_dim)
    
    prior1_cc = Model(cnn_unit_cc.input, cnn_unit_cc.output, name='CC_prior1')([input_cc_x1,input_cc_x2])
    prior2_cc = Model(cnn_unit_cc.input, cnn_unit_cc.output, name='CC_prior2')([input_cc_x3,input_cc_x4])
    prior3_cc = Model(cnn_unit_cc.input, cnn_unit_cc.output, name='CC_prior3')([input_cc_x5,input_cc_x6])
    prior4_cc = Model(cnn_unit_cc.input, cnn_unit_cc.output, name='CC_prior4')([input_cc_x7,input_cc_x8])
    
    
    x = concatenate([prior4_cc,prior3_cc,prior2_cc,prior1_cc], axis = 1,  name = 'CC_concat')
    
    CC_output = GRU(latent_dim, input_shape=(4, 512))(x)

    ############################################################################################################
    ############################################################################################################
    # MLO_views 
    ############################################################################################################
    input_mlo_x1 = Input(shape=input_image_dim, name = 'MLO_input1')
    input_mlo_x2 = Input(shape = input_image_dim, name = 'MLO_input2')
    
    input_mlo_x3 = Input(shape=input_image_dim, name = 'MLO_input3')
    input_mlo_x4 = Input(shape = input_image_dim, name = 'MLO_input4')
    
    input_mlo_x5 = Input(shape=input_image_dim, name = 'MLO_input5')
    input_mlo_x6 = Input(shape = input_image_dim, name = 'MLO_input6')
    
    input_mlo_x7 = Input(shape=input_image_dim, name = 'MLO_input7')
    input_mlo_x8 = Input(shape = input_image_dim, name = 'MLO_input8')
    
    cnn_unit_mlo = CNN_MODEL(input_image_dim) 
    
    
    prior1_mlo = Model(cnn_unit_mlo.input, cnn_unit_mlo.output, name='MLO_prior1')([input_mlo_x1,input_mlo_x2])
    prior2_mlo = Model(cnn_unit_mlo.input, cnn_unit_mlo.output, name='MLO_prior2')([input_mlo_x3,input_mlo_x4])
    prior3_mlo = Model(cnn_unit_mlo.input, cnn_unit_mlo.output, name='MLO_prior3')([input_mlo_x5,input_mlo_x6])
    prior4_mlo = Model(cnn_unit_mlo.input, cnn_unit_mlo.output, name='MLO_prior4')([input_mlo_x7,input_mlo_x8])
    
    x = concatenate([prior4_mlo,prior3_mlo,prior2_mlo,prior1_mlo], axis = 1,  name = 'MLO_concat')

    MLO_output = GRU(latent_dim, input_shape=(4,512))(x)


    ############################################################################################################
    ############################################################################################################
    # Concatenation of MLO&CC
    ############################################################################################################
    #In case of under-fitting or overfitting change or remove dropout or regularizer
    mergedOut = concatenate([CC_output, MLO_output])
    mergedOut = Flatten()(mergedOut)    
    mergedOut = Dense(128, activation='relu', kernel_regularizer = l2(0.001))(mergedOut)
    # mergedOut = Dropout(.5)(mergedOut)
    mergedOut = Dense(32, activation='relu', kernel_regularizer = l2(0.001) )(mergedOut)
    # mergedOut = Dropout(.35)(mergedOut)
    mergedOut = Dense(2, activation='softmax', name ='predictions')(mergedOut)
     
    model = Model([input_cc_x1,input_cc_x2,input_cc_x3, input_cc_x4, input_cc_x5, input_cc_x6, input_cc_x7,input_cc_x8,input_mlo_x1,input_mlo_x2,input_mlo_x3, input_mlo_x4, input_mlo_x5, input_mlo_x6, input_mlo_x7,input_mlo_x8], mergedOut, name='cnn_gru_ff')
    
    OPTIMIZER= RMSprop(lr = 0.01)
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics = metrics)
    
    return model

