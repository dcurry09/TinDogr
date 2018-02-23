#######################################
# Transfer Learning with Keras
# Predicts Dog breed using VGG-16 as frozen model
# Data is 120 breeds woth 20K images
# by David Curry
#
# Feb. 2018
#######################################

# import dependencies
import csv as csv
import numpy as np
import pandas as pd
import pylab as py
import operator, re, progressbar, sys
import multiprocessing
from collections import Counter
import matplotlib.pyplot as plt
from operator import itemgetter
import pickle, logging
from skimage import color, exposure, transform, io
from time import time
import codecs, glob
from tempfile import TemporaryFile
import os

from keras.utils import to_categorical
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.engine.topology import Input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

# My modules
sys.path.insert(0,"/Users/HAL3000/Dropbox/coding/my_modules/")
import keras_modules as my_keras_modules
import misc_modules as misc

##################################################

# Paths
root_dir = '/Users/HAL3000/Dropbox/coding/Insight/Tinder/data/dog_breeds/Images/'
train_data_dir = root_dir+'/train/'
test_data_dir  = root_dir+'/test/'
top_model_weights_path = 'weights/my_model'

# Dataset Constants
NUM_CLASSES = 120
IMG_SIZE = 64

# Network Inputs
train_samples = 14458
test_samples  = 6122
#k.set_image_dim_ordering('tf')
batch_size = 10
epochs = 50

##################################################


def get_class(img_path):     
    '''Returns class labels as ints from dir names'''
    temp = img_path.split('/')[-2]
    return int(re.sub('[^0-9]','', temp))

 
def save_class_labels():
   ''' Saves test/train labels for later use
   '''

   print('\nSaving Class Labels')

   train_labels = []
   test_labels  = []
   
   train_path = glob.glob(os.path.join(train_data_dir, '*/*.jpg'))
   test_path  = glob.glob(os.path.join(test_data_dir, '*/*.jpg'))
   
   for class_dir in os.listdir(train_data_dir):
       print('\nClass Dir:', class_dir)
       for i,img_path in enumerate(os.listdir(train_data_dir+class_dir)):
           #print(img_path)
           #label = get_class(img_path)
           if i==1:
               print('---> Processing Image:',img_path)
           #    print('Saving Class Label:', label)
           #train_labels.append(label)    
        
   for i,img_path in enumerate(test_path):
       label = get_class(img_path)
       if i==1:
           print('---> Processing Image:',img_path)
           print('Saving Class Label:', label)
       test_labels.append(label)     


   # one hot encode
   train_y = to_categorical(train_labels, NUM_CLASSES)
   test_y  = to_categorical(test_labels, NUM_CLASSES)

   #print(train_labels, test_y)
   print('Test Y Shape:', train_y.shape)
   
   pickle.dump([train_y, test_y], open('preprocess.p','wb'))
   print('Data converted into train, test and validation')

   return train_y, test_y, train_labels, test_labels


def save_bottleneck_features():
        ''' Saves bottleneck features for test/train sets for use later on.
            Separating this step speeds up later optimization.
        '''
        print('\n Saving Bottleneck Features...')
   
        model = applications.VGG16(weights = 'imagenet', include_top = False)

        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range = 40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            fill_mode = 'nearest'
         )

        train_generator = datagen.flow_from_directory(
           train_data_dir,
           target_size = (IMG_SIZE, IMG_SIZE),
           batch_size = batch_size,
           class_mode = None,
           shuffle = False
        )

        if not os.path.isfile('weights/bottleneck_features_train.npy'):
            bottleneck_features_train = model.predict_generator(train_generator, train_samples//batch_size, verbose=1)
            np.save(open('weights/bottleneck_features_train.npy','wb'), bottleneck_features_train)
        else:
            bottleneck_features_train = np.load('weights/bottleneck_features_train.npy')

        test_generator = datagen.flow_from_directory(
           test_data_dir,
           target_size = (IMG_SIZE, IMG_SIZE),
           batch_size = batch_size,
           class_mode = None,
           shuffle = False
        )
            
        if not os.path.isfile('weights/bottleneck_features_test.npy'):
            bottleneck_features_test = model.predict_generator(test_generator, test_samples//batch_size, verbose=1)
            np.save(open('weights/bottleneck_features_test.npy','wb'), bottleneck_features_test)
        else:
            bottleneck_features_test = np.load('weights/bottleneck_features_test.npy')
            
        
        
        print('\nData Class Indices from Gen:', train_generator.class_indices)
        print('\nData Classes from Gen:', train_generator.classes)
        
        train_y = to_categorical(train_generator.classes)
        test_y = to_categorical(test_generator.classes)
        train_labels = train_generator.class_indices
        np.save(open('weights/train_y.npy','wb'), train_y)
        np.save(open('weights/train_labels.npy','wb'), train_labels)
        np.save(open('weights/test_y.npy','wb'), test_y)
        
        return bottleneck_features_train, bottleneck_features_test, train_y, train_labels, test_y
        

def train_top_model(train_data, train_Y, test_data, test_Y):
   ''' Training of FC layers with bottleneck features as inputs
   '''

   print('\n Training the FC Layers...')
   
   model = Sequential()
   model.add(Flatten(input_shape = train_data.shape[1:]))
   model.add(Dense(2056, activation='relu'))
   model.add(Dropout(0.2))
   model.add(Dense(1028, activation='relu'))
   model.add(Dense(NUM_CLASSES, activation='softmax'))

   opt = optimizers.SGD(lr=0.01)
   model.compile(optimizer = opt,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

   checkpointer = ModelCheckpoint(filepath='model.best.hdf5', verbose=1, save_best_only=False)
   
   model.fit(train_data, train_Y,
             epochs=epochs,
             batch_size=batch_size,
             validation_data = [test_data, test_Y],
             callbacks = [checkpointer])

   model.save_weights(top_model_weights_path)
   
   
###########################################################################3

if __name__ == "__main__":

    #train_y, test_y, train_labels, test_labels = save_class_labels()
    
    if (not os.path.isfile('weights/bottleneck_features_train.npy')) or (not os.path.isfile('weights/bottleneck_features_test.npy')):
        train_data, test_data, train_y, train_labels, test_y  = save_bottleneck_features()
    else:
        print('\nLoading Bottleneck Features...')
        train_data = np.load('weights/bottleneck_features_train.npy')
        train_data = np.load('weights/bottleneck_features_test.npy')
        train_y = np.load('weights/train_y.npy')
        test_y = np.load('weights/test_y.npy') 
        train_labels = np.load('weights/train_labels.npy')
        print('\nData Class Indices from Gen:', train_labels)
        print('\nData Classes from Gen:', train_y)
        print('Shape of train Y:', train_y.shape)
        
        
    train_top_model(train_data, train_y, test_data, test_y)



    
