import tensorflow as tf

print(tf.__version__)

from keras.layers                import *
from keras.preprocessing.image   import ImageDataGenerator
from keras.utils                 import to_categorical
from keras.optimizers            import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Input
from keras.applications.vgg16 import VGG16

from keras.models import Sequential, Model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn
import scipy
# from skimage.transform import resize
import csv
# from tqdm import tqdm
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

#Clasificamos las imagenes
bs = 32 #bach size
k = 2
num_classes = 29
# Generador de imágenes de entrenamiento.
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=(0.3),
        zoom_range=(0.3),
        width_shift_range=(0.2),
        height_shift_range=(0.2),
        validation_split = 0.2,
        brightness_range=(0.05,0.85),
        horizontal_flip=False,
        rotation_range=20
)

# Carga de imágenes al generador de entrenamiento desde directorio.
train_generator = train_datagen.flow_from_directory(
        './train',
        class_mode='categorical',
        shuffle=True,
        target_size=(28, 28),
        color_mode = 'rgb', 
        subset='training',
        batch_size=bs)

valid_generator = train_datagen.flow_from_directory(
        './train',
        class_mode='categorical',
        shuffle=True,
        target_size=(28, 28),
        color_mode = 'rgb', 
        subset='validation',
        batch_size=bs)

# print(train_generator, valid_generator)

print(train_generator.class_indices)

training_number=69600
test_number=17400

image_input=Input(shape=(28, 28, 3))

# ## MODEL TO TRAINING
# model2 = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
# model2.load_weights('./vgg16_weights_tf_dim_ordering_tf_kernels.h5')
# model2.summary()

#Creación de modelo 
model = Sequential()

inputShape = (28, 28, 3)
model.add(Conv2D(32,(3,3), input_shape=inputShape))
# model.add(Conv2D(32,(3,3)))
model.add(MaxPool2D())
model.add(Conv2D(64,(3,3), input_shape=inputShape))
# model.add(Conv2D(32,(3,3)))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(num_classes,activation='softmax', name='output'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model_history = model.fit(  
    train_generator,
    epochs=30,
    validation_data=valid_generator,
    steps_per_epoch=training_number//bs,
    validation_steps=test_number//bs)

model.save('model_second.h5')