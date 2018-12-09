#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tommdg
"""
import numpy as np
import os
from keras_preprocessing.image import DataFrameIterator, load_img, img_to_array, array_to_img


def load_rgb_img(path, _id, channels=None, target_size=None, interpolation='nearest'):
    if channels is None:
        channels = ['red', 'green', 'blue', 'yellow']
    img = [img_to_array(load_img(path + '{}_{}.png'.format(_id, suffix),
                                 color_mode='grayscale',
                                 target_size=target_size,
                                 interpolation=interpolation).astype(np.float32)/255)
           for suffix in channels]
    return img


def rgb_img_to_array(img, data_format='channels_last'):
    if data_format == 'channels_last':
        return np.stack(img, axis=-1)
    elif data_format == 'channels_first':
        return np.stack(img, axis=0)


class RGBDataFrameIterator(DataFrameIterator):

    """Custom DataFrameIterator to support channels provided in seperated files. """

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=self.dtype)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            if self.directory is not None:
                img_path = os.path.join(self.directory, fname)
            else:
                img_path = fname
            img = load_rgb_img(img_path,
                               target_size=self.target_size,
                               interpolation=self.interpolation)
            x = rgb_img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            for i in img:
                if hasattr(img, 'close'):
                    img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(self.dtype)
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=self.dtype)
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'other':
            batch_y = self.data[index_array]
        else:
            return batch_x
        return batch_x, batch_y
