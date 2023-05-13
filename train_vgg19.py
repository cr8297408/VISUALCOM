import tensorflow as tf

print(tf.__version__)

from keras.layers                import *
from keras.preprocessing.image   import ImageDataGenerator
from keras.utils                 import to_categorical
# from keras.optimizers            import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Input
from keras.applications.vgg19 import VGG19
from keras import regularizers
# from keras.models import Sequential, Model
#Agregamos los callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.optimizers import SGD
# from sklearn.utils import class_weight

import matplotlib.pyplot as plt
# import random
# import cv2
# import pandas as pd
# import numpy as np
# import matplotlib.gridspec as gridspec
# import seaborn as sns
# import sklearn
# import scipy
# # from skimage.transform import resize
# import csv
# from tqdm import tqdm
# from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import confusion_matrix

## deshabilitar ssl for vgg19 model download
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


#variables
bs = 64 #bach size
k = 2
num_classes = 29
epochs = 25

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
  target_size=(28*k, 28*k),
  color_mode = 'rgb', 
  subset='training',
  batch_size=bs
)

valid_generator = train_datagen.flow_from_directory(
  './train',
  class_mode='categorical',
  shuffle=True,
  target_size=(28*k, 28*k),
  color_mode = 'rgb', 
  subset='validation',
  batch_size=bs
)

# print(train_generator, valid_generator)

print(train_generator.class_indices)

training_number=69600
test_number=28

model = VGG19()
model.summary()

VGG19_model = VGG19(input_shape=(28*k,28*k,3), include_top=False, weights='imagenet')

print(len(VGG19_model.layers))

for layer in VGG19_model.layers[:6]:
  layer.trainable = False
  
# Creamos un nuevo modelo vacio.
model = tf.keras.Sequential()

# Añadimos el modelo preentrenado como si se tratase de una capa.
model.add(VGG19_model)

# Continuamos añadiendo más capas que sí serán entrenadas...
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01), activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss="categorical_crossentropy", optimizer= SGD(learning_rate=0.01), metrics=['accuracy'])

#este primer callback guarda el modelo en el momento en que obtiene su mayor precision
checkpointer = ModelCheckpoint(filepath='model', verbose=1, save_best_only=True, monitor = 'val_acc', mode = 'max')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)
#Procedemos a entrenar
history= model.fit(train_generator,
  validation_data = valid_generator, 
  callbacks=[reduce_lr, checkpointer], 
  epochs=epochs
)

model.save('model_with_vgg19.h5');
