#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tommdg
"""

##################
### IMPORTS AND UTILS
##################

# System
import gc
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf

# Linear algebra and ML tools
import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", palette='husl')


INPUT_PATH = '../input/human-protein-atlas-image-classification/'
TRAIN_PATH = INPUT_PATH + 'train/'
WEIGHT_PATH = '../input/multilabel-cnn-green-1/multilabel_cnn_green.h5'
label_mapping = {
    'class_0': 'Nucleoplasm',
    'class_1': 'Nuclear membrane',
    'class_2': 'Nucleoli',
    'class_3': 'Nucleoli fibrillar center',
    'class_4': 'Nuclear speckles',
    'class_5': 'Nuclear bodies',
    'class_6': 'Endoplasmic reticulum',
    'class_7': 'Golgi apparatus',
    'class_8': 'Peroxisomes',
    'class_9': 'Endosomes',
    'class_10': 'Lysosomes',
    'class_11': 'Intermediate filaments',
    'class_12': 'Actin filaments',
    'class_13': 'Focal adhesion sites',
    'class_14': 'Microtubules',
    'class_15': 'Microtubule ends',
    'class_16': 'Cytokinetic bridge',
    'class_17': 'Mitotic spindle',
    'class_18': 'Microtubule organizing center',
    'class_19': 'Centrosome',
    'class_20': 'Lipid droplets',
    'class_21': 'Plasma membrane',
    'class_22': 'Cell junctions',
    'class_23': 'Mitochondria',
    'class_24': 'Aggresome',
    'class_25': 'Cytosol',
    'class_26': 'Cytoplasmic bodies',
    'class_27': 'Rods & rings',
}

##################
### Preprocessing
##################

# loading
n_classes = len(label_mapping)


def vectorize(arr, size=n_classes):
    res = np.zeros(size, dtype=int)
    res[arr] = 1
    return res


train_labels = pd.read_csv(INPUT_PATH + 'train.csv')
train_labels['Target'] = train_labels['Target'].apply(lambda x: list(map(int, x.split(' '))))
train_labels['target_v'] = train_labels['Target'].apply(vectorize)
train_labels[list(label_mapping.keys())] = pd.DataFrame(train_labels['target_v'].values.tolist(),
                                                        index=train_labels.index)
train_labels['Id'] = train_labels['Id'] + '_green.png'
class_count = train_labels['target_v'].sum()

#  Split
from sklearn.model_selection import StratifiedShuffleSplit

def coalesce(arr):
    return arr[np.argmin(np.array(class_count)[arr])]

train_labels['y_coal'] = train_labels['Target'].apply(coalesce)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, val_index in sss.split(np.zeros(len(train_labels)), train_labels['y_coal']):
    df_train = train_labels.iloc[train_index]
    df_val = train_labels.iloc[val_index]

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)

#  some gc collection
del train_labels
gc.collect()


##################
###     Model
##################

# constants and utils
batch_size_train = 64
batch_size_val = 64
WIDTH = 256
HEIGHT = 256
channels = 1

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

# datagenerator
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_dataframe(dataframe=df_train, directory=TRAIN_PATH, x_col="Id", y_col=list(label_mapping.keys()), color_mode='grayscale',
                                              class_mode='other', target_size=(WIDTH,HEIGHT), batch_size=batch_size_train)
val_generator = datagen.flow_from_dataframe(dataframe=df_val, directory=TRAIN_PATH, x_col="Id", y_col=list(label_mapping.keys()), color_mode='grayscale',
                                            class_mode="other", target_size=(WIDTH,HEIGHT), batch_size=batch_size_val)

# CNN architecture
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

filters0 = 8
nb_dense = 64

model = Sequential()
model.add(Conv2D(filters0, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(WIDTH, HEIGHT, channels)))
model.add(Conv2D(filters0, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(WIDTH, HEIGHT, channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters0 * 2, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters0 * 2, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters0 * 4, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters0 * 4, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters0 * 4, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters0 * 4, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters0 * 8, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters0 * 8, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters0 * 8, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters0 * 16, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters0 * 16, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters0 * 16, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(nb_dense, activation='relu'))
model.add(Dense(n_classes, activation='sigmoid'))

# Optimizers and callbacks
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = val_generator.n // val_generator.batch_size

epochs = 100
learning_rate = 0.0001
callback= EarlyStopping(monitor='val_loss', patience=30)

model.load_weights(WEIGHT_PATH)
model.compile(loss=f1_loss,
              optimizer=Adam(lr=learning_rate),
              metrics=[f1])

output = model.fit_generator(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=STEP_SIZE_VALID,
    callbacks=[callback],
    use_multiprocessing=True,
    )

model.save_weights('multilabel_cnn_green_200.h5')

# data visualisation
## loss
fig, ax = plt.subplots(figsize=(11,6))
ax.plot(output.epoch, output.history['loss'], label='train')
ax.plot(output.epoch, output.history['val_loss'], label='val')
ax.set_title('Training and validation performance')
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend()
fig.savefig('loss_100_200e.png')

## f1 score
fig, ax = plt.subplots(figsize=(11,6))
ax.plot(output.epoch, output.history['f1'], label='train')
ax.plot(output.epoch, output.history['val_f1'], label='val')
ax.set_title('Training and validation performance')
ax.set_ylabel('macro f1 score')
ax.set_xlabel('epoch')
ax.legend()
fig.savefig('metrics_100_200e.png')
