#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tommdg
"""

from functools import partial
import numpy as np
import pandas as pd

from keras.models import Sequential, load_model, Model
from keras.layers import (Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization,
                          Concatenate, ReLU)
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", palette='colorblind')


INPUT_PATH = '../input/'
TRAIN_PATH = INPUT_PATH + 'train/'
LABEL_MAPPING = {
    0: 'nucleoplasm',
    1: 'nuclear_membrane',
    2: 'nucleoli',
    3: 'nucleoli_fibrillar_center',
    4: 'nuclear_speckles',
    5: 'nuclear_bodies',
    6: 'endoplasmic_reticulum',
    7: 'golgi_apparatus',
    8: 'peroxisomes',
    9: 'endosomes',
    10: 'lysosomes',
    11: 'intermediate_filaments',
    12: 'actin_filaments',
    13: 'focal_adhesion_sites',
    14: 'microtubules',
    15: 'microtubule_ends',
    16: 'cytokinetic_bridge',
    17: 'mitotic_spindle',
    18: 'microtubule_organizing_center',
    19: 'centrosome',
    20: 'lipid_droplets',
    21: 'plasma_membrane',
    22: 'cell_junctions',
    23: 'mitochondria',
    24: 'aggresome',
    25: 'cytosol',
    26: 'cytoplasmic_bodies',
    27: 'rods_and_rings',
}

# HYPERPARAMETERS
WIDTH = 512
HEIGHT = 512
CHANNELS = 1
POS_SAMPLING = 1
NEG_SAMPLING = 1
USE_BASELINE_MODEL = True
NB_METRIC_SAMPLES = 2000
SEED = 42
np.random.seed(SEED)
tf.set_random_seed(SEED)

# MODEL PARAMETERS
BATCH_SIZE = 32
DEPTH = 6
NB_CONV = 2
NB_FILTER_INIT = 8
NB_DENSE_SIZE = 32
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)


def f1(y_true, y_pred, threshold):
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def create_baseline_model(input_shape=(WIDTH, HEIGHT, CHANNELS)):
    dropRate = 0.25

    init = Input(input_shape)
    x = BatchNormalization(axis=-1)(init)
    x = Conv2D(8, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(8, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(16, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    c1 = Conv2D(16, (3, 3), padding='same')(x)
    c1 = ReLU()(c1)
    c2 = Conv2D(16, (5, 5), padding='same')(x)
    c2 = ReLU()(c2)
    c3 = Conv2D(16, (7, 7), padding='same')(x)
    c3 = ReLU()(c3)
    c4 = Conv2D(16, (1, 1), padding='same')(x)
    c4 = ReLU()(c4)
    x = Concatenate()([c1, c2, c3, c4])
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(32, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(64, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(128, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(28)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.1)(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = Model(init, x)
    return model


class BinaryConvolutionNeuralNet(object):

    """Docstring for BinaryConvolutionNeuralNet. """

    def __init__(self, label, neg_sampling=NEG_SAMPLING, pos_sampling=POS_SAMPLING, test_size=0.2, augmentation=False,
                 batch_size=BATCH_SIZE, depth=DEPTH, nb_conv=NB_CONV, nb_filter=NB_FILTER_INIT,
                 nb_dense_size=NB_DENSE_SIZE, kernel_size=KERNEL_SIZE, pool_size=POOL_SIZE,
                 load_pretrained_model=False, use_baseline_model=USE_BASELINE_MODEL, verbose=True, seed=SEED):
        self.label = label
        self.name = LABEL_MAPPING[label]
        self.verbose = verbose
        self.seed = seed
        self.train_generator, self.val_generator = self._create_train_val_generator(neg_sampling, pos_sampling,
                                                                                    test_size, augmentation,
                                                                                    batch_size)
        self.model = self._create_binary_cnn(use_baseline_model, load_pretrained_model,
                                             depth, nb_conv, nb_filter, nb_dense_size, kernel_size, pool_size)

    def _create_train_val_generator(self, neg_sampling, pos_sampling, test_size, augmentation, batch_size):
        df_labels = self._create_df_labels(neg_sampling, pos_sampling)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
        for train_index, val_index in sss.split(np.zeros(len(df_labels)), df_labels['y']):
            df_train = df_labels.iloc[train_index].reset_index(drop=True)
            df_val = df_labels.iloc[val_index].reset_index(drop=True)

        if self.verbose:
            n_pos = df_train['y'].sum()
            print('Train size: {} rows including {} positive samples ({:.1%})'.format(len(df_train), n_pos,
                                                                                      n_pos / len(df_train)))
            n_pos = df_val['y'].sum()
            print('Val size: {} rows including {} positive samples ({:.1%})'.format(len(df_val), n_pos,
                                                                                    n_pos / len(df_val)))

        data_generator = self._create_data_generator(augmentation)

        train_generator = data_generator.flow_from_dataframe(
            dataframe=df_train, directory=TRAIN_PATH, x_col="Id", y_col='y', shuffle=True, color_mode='grayscale',
            class_mode='binary', target_size=(WIDTH, HEIGHT), batch_size=batch_size,
        )
        val_generator = data_generator.flow_from_dataframe(
            dataframe=df_val, directory=TRAIN_PATH, x_col="Id", y_col='y', shuffle=True, color_mode='grayscale',
            class_mode="binary", target_size=(WIDTH, HEIGHT), batch_size=batch_size,
        )

        return train_generator, val_generator

    def _create_df_labels(self, neg_sampling=1, pos_sampling=1):
        df = pd.read_csv(INPUT_PATH + 'train.csv')
        df['Id'] = df['Id'] + '_green.png'  # to take one channel only into account
        df['y'] = df['Target'].apply(lambda x: self.label in list(map(int, x.split(' ')))).astype(np.int8)
        del df['Target']
        df_neg = df[df['y'] == 0]
        df_pos = df[df['y'] == 1]
        self.smart_threshold = len(df_pos) / len(df)
        self.f1 = partial(f1, threshold=self.smart_threshold)
        self.f1.__name__ = 'f1'
        if self.verbose:
            print('Threshold computed: {:.3f}'.format(self.smart_threshold))

        if neg_sampling < 1:
            n_rows = len(df_neg)
            indices = df_neg.sample(frac=1-neg_sampling, random_state=self.seed).index
            df_neg = df_neg.drop(indices)
            if self.verbose:
                print('Negative sampling: keeping {} out of {} rows'.format(len(df_neg), n_rows))

        if pos_sampling != 1:
            n_rows = len(df_pos)
            if pos_sampling > 1:
                df_pos_sample = pd.concat([df_pos]*int(pos_sampling), ignore_index=True)
                rest = pos_sampling - int(pos_sampling)
                if rest > 0:
                    indices = df_pos.sample(frac=1-pos_sampling, random_state=self.seed).index
                    df_pos = df_pos.drop(indices)
                df_pos = pd.concat([df_pos, df_pos_sample], ignore_index=True)
            else:
                indices = df_pos.sample(frac=1-pos_sampling, random_state=self.seed).index
                df_pos = df_pos.drop(indices)
            if self.verbose:
                print('Positive sampling: keeping {} out of {} rows'.format(len(df_pos), n_rows))

        return shuffle(pd.concat([df_pos, df_neg], ignore_index=True), random_state=self.seed)

    def _create_data_generator(self, augmentation=False):
        if augmentation:
            kwargs = {'rotation_range': 90}
        else:
            kwargs = {}
        return ImageDataGenerator(rescale=1./255, **kwargs)

    def _create_binary_cnn(self, use_baseline_model, load_pretrained_model, depth, nb_conv, nb_filter, nb_dense_size,
                           kernel_size, pool_size):
        if load_pretrained_model:
            model = load_model('base.model', custom_objects={'f1': self.f1})
        elif use_baseline_model:
            model = create_baseline_model()
        else:
            if isinstance(nb_conv, int):
                nb_conv = [nb_conv]*depth
            if isinstance(nb_filter, int):
                nb_filter = [nb_filter * 2**k for k in range(depth)]

            model = Sequential()
            for k in range(depth):
                for _ in range(nb_conv[k]):
                    if k == 0:
                        kwargs = {'input_shape': (WIDTH, HEIGHT, CHANNELS)}
                    else:
                        kwargs = {}
                    model.add(Conv2D(nb_filter[k],
                                     kernel_size=kernel_size,
                                     padding='same',
                                     activation='relu',
                                     **kwargs))
                model.add(MaxPooling2D(pool_size=pool_size))
            model.add(Flatten())
            model.add(Dense(nb_dense_size, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

        if self.verbose:
            model.summary()

        return model

    def fit(self, epochs=100, patience=30, learning_rate=0.001, save_model=True):
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=[self.f1])

        callbacks = []
        callbacks.append(EarlyStopping(monitor='val_loss', patience=patience))
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=int(self.verbose),
                                           mode='min'))
        if save_model:
            model_name = '{}_binary_cnn.model'.format(self.name)
            callbacks.append(ModelCheckpoint(model_name, monitor='val_loss', verbose=int(self.verbose),
                                             save_best_only=True, save_weights_only=False, mode='min', period=1))

        self.history = self.model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=len(self.val_generator),
            callbacks=callbacks,
        )

    def save_history_metrics(self):
        #  Loss
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(self.history.epoch, self.history.history['loss'], label='train')
        ax.plot(self.history.epoch, self.history.history['val_loss'], label='val')
        ax.set_title('Training and validation performance')
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend()
        fig.savefig('{}_loss_history.png'.format(self.name))

        #  F1 score
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(self.history.epoch, self.history.history['f1'], label='train')
        ax.plot(self.history.epoch, self.history.history['val_f1'], label='val')
        ax.set_title('Training and validation performance')
        ax.set_ylabel('macro f1 score')
        ax.set_xlabel('epoch')
        ax.set_ylim((0, 1))
        ax.legend()
        fig.savefig('{}_f1_history.png'.format(self.name))

    def save_prediction_metrics(self):
        # TODO: compute predictions on real dataset, not downsampled nor upsampled!!
        y_pred = []
        y_true = []
        for _ in range(NB_METRIC_SAMPLES // self.val_generator.batch_size + 1):
            X, y = next(self.val_generator)
            y_true.append(y)
            y_pred.append(self.model.predict_on_batch(X))
        y_true = np.concatenate(y_true, axis=None)
        y_pred = np.concatenate(y_pred, axis=None)

        # sorted predictions
        fig, ax = plt.subplots(figsize=(11, 6))
        indices = np.argsort(y_pred)
        ax.scatter(range(len(y_pred)), y_pred[indices],
                   c=list(map(lambda x: ['r', 'b'][int(x)], y_true[indices])), marker='.')
        ax.set_title('Predictions for {} samples'.format(len(y_pred)))
        ax.set_ylabel('y_hat')
        ax.set_ylim((0, 1))
        fig.savefig('{}_predictions.png'.format(self.name))

        # prediction distribution
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.distplot(y_pred, kde=True, hist=False, ax=ax, kde_kws={'shade': True})
        ax.set_title('Predictions distribution for {} samples'.format(len(y_pred)))
        fig.savefig('{}_predictions_distribution.png'.format(self.name))
