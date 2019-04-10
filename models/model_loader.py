"""
Applications for image classifcation.
Each function returns a pretrained ImageNet model, which is return logits.
The models are based on keras.applications/tensorflow hub models and
contain additionally pretrained patterns.
"""
# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
import wget
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, \
    Dropout, Flatten, Conv2D, MaxPooling2D, \
    Concatenate, Input, Lambda, merge, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from io import open
import six
# End: Python 2/3 compatability header small

###############################################################################
###############################################################################
###############################################################################

import keras
from keras import layers
import keras_applications
import keras.backend as K
from keras.preprocessing import image
import keras.utils.data_utils
import numpy as np
import warnings
import os

import tensorflow as tf
from scipy.special import softmax
import cv2

#import models.inceptionV4.inception_v4 as inception_v4
import models.model_utils as mutils
import innvestigate.utils as iutils

__all__ = [
    "AlexNet",
    "VGG16",
    "VGG19",
    "Inception_v1",
    "Inception_v2",
    "Inception_v3",
    "Inception_v4",
    "Inception_Resnet_v2",
    "Resnet_v1_50",
    "Resnet_v1_101",
    "Resnet_v1_152",
    "Resnet_v2_50",
    "Resnet_v2_101",
    "Resnet_v2_152",
    "ResNeXt_50",
    "ResNeXt_101",
]

###############################################################################
###############################################################################
###############################################################################

class Hub_Model:
    def __init__(self):
        pass

    def preprocess_input(self, x, **kwargs):
        """
        Prepocesses singe input image and returns batch

        Args:
            x: RGB image or image batch
            **kwargs:

        Returns:
            img batch

        """

        def _preprocess(x):
            x = cv2.resize(x, dsize=self.get_image_size())
            x = np.array(x)
            x = x / np.amax(x)  # normalize input
            return x

        if not isinstance(x,np.ndarray):
            x = np.array(x)

        if x.ndim == 3:
            x = _preprocess(x,**kwargs)
            x = x[np.newaxis, ...]
            return x

        for i in range(x.shape[0]):
            x[i] = _preprocess(x[i],**kwargs)
        return x

    def predict_wo_softmax(self,x, batch_size=None, verbose=0, steps=None):
        return self._classifier_model.predict(x, batch_size=None, verbose=0, steps=None)

    def predict_with_softmax(self,x, batch_size=None, verbose=0, steps=None):
        preds = self.predict_wo_softmax(x, batch_size=None, verbose=0, steps=None)
        return softmax(preds)

    def decode_predictions(self, preds, top=5, **kwargs):
        return mutils.decode_predictions(preds, top, **kwargs)

    def get_model(self):
        return self.classifier_model

    def get_color_coding(self):
        return self.color_coding

    def get_image_size(self):
        return self._classifier_model.layers[0].input_shape[1:3]

class Keras_App_Model:
    def __init__(self):
        pass

    def preprocess_input(self, x, **kwargs):
        """
            Prepocesses singe input image and returns batch

            Args:
                x: RGB image or image batch
                **kwargs:

            Returns:
                img batch
        """

        def _preprocess(x):
            return self._module.preprocess_input(x, **kwargs)

        if not isinstance(x,np.ndarray):
            x = np.array(x)

        if x.ndim == 3:
            x = cv2.resize(x,self.get_image_size())
            x = _preprocess(x[np.newaxis,...],**kwargs)
            return x

        x = _preprocess(x)
        return x

    def predict_wo_softmax(self,x, batch_size=None, verbose=0, steps=None):
        return self._classifier_model.predict(x, batch_size=None, verbose=0, steps=None)

    def predict_with_softmax(self,x, batch_size=None, verbose=0, steps=None):
        preds = self.predict_wo_softmax(x, batch_size=None, verbose=0, steps=None)
        return softmax(preds)

    def decode_predictions(self, preds, top=5, **kwargs):
        return self._module.decode_predictions(preds, top, **kwargs)

    def get_image_size(self):
        return self._classifier_model.layers[0].input_shape[1:3]

###############################################################################
###############################################################################
###############################################################################

class AlexNet:
    """
        Wrapper for AlexNet based on implementation of
        University of Toronto.

        Warning: This net does not work yet

        (http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)
    """

    _IMAGE_SIZE = (227,227)
    __color_coding = 'bgr'
    __patterns = None

    def __init__(self, download_location, include_top=True, weights='imagenet'):
        def get_alexNet_model_wo_softmax(x, include_top=True):
            def convLayer(x, name, filters, kernel_size=(3, 3), strides=(1, 1)
                          , padding="valid", groups=1):
                """(grouped) convolution"""
                if groups == 1:
                    y = Conv2D(filters=filters, kernel_size=kernel_size,
                               strides=(strides), padding=padding)(x)
                    y = Activation('relu')(y)
                    return y

                x_new = []
                channels = int(x.get_shape()[-1])
                split = int(channels / groups)
                split_upper = split
                split_lower = 0
                for i in range(groups):
                    x_new.append(Lambda(lambda x: x[:, :, :, split_lower:split_upper])(x))
                    split_lower = split_lower + split_upper
                    split_upper = split_upper + split_upper
                featureMaps = [Conv2D(filters=int(filters / groups), kernel_size=kernel_size,
                                      strides=(strides), padding=padding, name=str(name)+'_' + str(i + 1))(x_n) for i,x_n in enumerate(x_new)]

                # y = tf.concat(axis=3, values=featureMaps)
                y = Concatenate(axis=-1, name=name)(featureMaps)
                y = Activation('relu')(y)
                return y

            def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
                """
                This is the function used for cross channel normalization in the original
                Alexnet
                """

                def f(X):
                    b, ch, r, c = X.shape
                    half = n // 2
                    square = K.square(X)
                    extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1))
                                                          , (0, half))
                    extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
                    scale = k
                    for i in range(n):
                        scale += alpha * extra_channels[:, i:i + ch, :, :]
                    scale = scale ** beta
                    return X / scale

                return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)

            conv_1 = Conv2D(96, 11, 11, subsample=(4, 4), activation='relu',
                                   name='conv_1')(x)

            conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
            conv_2 = crosschannelnormalization()(conv_2)
            conv_2 = ZeroPadding2D((2, 2))(conv_2)
            conv_2 = convLayer(conv_2, name='conv_2', filters=128, kernel_size=(5,5),padding="same",groups=2)

            conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
            conv_3 = BatchNormalization()(conv_3)
            conv_3 = ZeroPadding2D((1, 1))(conv_3)
            conv_3 = Conv2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

            conv_4 = ZeroPadding2D((1, 1))(conv_3)
            conv_4 = convLayer(conv_4, name='conv_4', filters=192, kernel_size=(3, 3), padding="same", groups=2)

            conv_5 = ZeroPadding2D((1, 1))(conv_4)
            conv_5 = convLayer(conv_5, name='conv_5', filters=128, kernel_size=(3, 3), padding="same", groups=2)

            y = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

            if include_top:
                dense_1 = Flatten(name='flatten')(y)
                dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
                dense_2 = Dropout(0.5)(dense_1)
                dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
                dense_3 = Dropout(0.5)(dense_2)
                dense_3 = Dense(1000, name='dense_3')(dense_3)
                y = Activation('softmax', name='softmax')(dense_3)

            #model = Model(inputs=[x], outputs=[y])

            return y

        input = layers.Input(shape=list(self._IMAGE_SIZE) + [3])
        network_output = get_alexNet_model_wo_softmax(input, include_top=include_top)
        self._classifier_model = keras.models.Model(
            inputs=[input],
            outputs=[network_output]
        )

        if weights == 'imagenet':
            url = 'http://files.heuritech.com/weights/alexnet_weights.h5'
            if not os.path.isfile(download_location+'/alexnet_weights.h5'):
                wget.download(url, download_location)
            self._classifier_model.load_weights(download_location+'/alexnet_weights.h5')

    def preprocess_input(self, x, **kwargs):
        x = x.resize(self._IMAGE_SIZE)
        x = np.array(x).astype(np.float32)
        #convert to bgr
        x[:, :, 0], x[:, :, 2] = x[:, :, 2], x[:, :, 0]
        x = x - np.mean(x)
        x = x[np.newaxis, ...]
        return x

    def predict_wo_softmax(self,x, batch_size=None, verbose=0, steps=None):
        return self._classifier_model.predict(x, batch_size, verbose, steps)

    def predict_with_softmax(self,x, batch_size=None, verbose=0, steps=None):
        preds = self.predict_wo_softmax(x, batch_size, verbose, steps)
        return softmax(preds)

    def decode_predictions(self, preds, top=5, **kwargs):
        return mutils.decode_predictions_alexNet(preds, top, **kwargs)

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns

###############################################################################
###############################################################################
###############################################################################

class VGG16(Keras_App_Model):
    """
        Wrapper for VGG 16 based on Keras Applications
        (https://github.com/keras-team/keras/tree/master/keras/applications)
    """

    __name = 'vgg16'
    _module = keras.applications.vgg16
    __color_coding = 'rgb'#TODO innvestigate says bgr?
    __patterns = None

    def __init__(self, include_top=True, weights='imagenet', input_shape=(224,224,3)):
        classifier_model = keras.applications.vgg16.VGG16(include_top=include_top, weights=weights, input_shape=input_shape)
        if include_top:
            self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)
        else:
            self._classifier_model = classifier_model

        try:
            pattern_info = mutils._get_patterns_info(self.__name, "relu")
        except KeyError:
            warnings.warn("There are no patterns for network '%s'." % self.__name)
        else:
            patterns_path = keras.utils.data_utils.get_file(
                pattern_info["file_name"],
                pattern_info["url"],
                cache_subdir="innvestigate_patterns",
                hash_algorithm="md5",
                file_hash=pattern_info["hash"])
            patterns_file = np.load(patterns_path)
            patterns = [patterns_file["arr_%i" % i]
                        for i in range(len(patterns_file.keys()))]
            self.__patterns = patterns

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns

class VGG19(Keras_App_Model):
    """
        Wrapper for VGG 19 based on Keras Applications
        (https://github.com/keras-team/keras/tree/master/keras/applications)
    """

    __name = 'vgg19'
    _module = keras.applications.vgg19
    __color_coding = 'rgb'  # TODO innvestigate says bgr?
    __patterns = None

    def __init__(self, include_top=True, weights='imagenet'):
        classifier_model = keras.applications.vgg19.VGG19(include_top=include_top, weights=weights)
        if include_top:
            self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)
        else:
            self._classifier_model = classifier_model

        try:
            pattern_info = mutils._get_patterns_info(self.__name, "relu")
        except KeyError:
            warnings.warn("There are no patterns for network '%s'." % self.__name)
        else:
            patterns_path = keras.utils.data_utils.get_file(
                pattern_info["file_name"],
                pattern_info["url"],
                cache_subdir="innvestigate_patterns",
                hash_algorithm="md5",
                file_hash=pattern_info["hash"])
            patterns_file = np.load(patterns_path)
            patterns = [patterns_file["arr_%i" % i]
                        for i in range(len(patterns_file.keys()))]
            self.__patterns = patterns

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns

###############################################################################
###############################################################################
###############################################################################

class Resnet_v1_50(Keras_App_Model):
    """
    Wrapper for Resnet v1 50 based on keras_applications
    (https://github.com/keras-team/keras-applications)
    """

    __name = 'resnet v1 50'
    _module = keras_applications.resnet
    __color_coding = 'rgb'
    __patterns = None

    def __init__(self, include_top=True, weights='imagenet'):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model = keras_applications.resnet.ResNet50(include_top=include_top, weights=weights)
        if include_top:
            self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)
        else:
            self._classifier_model = classifier_model

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns


class Resnet_v1_101(Keras_App_Model):
    """
    Wrapper for Resnet v1 101 based on keras_applications
    (https://github.com/keras-team/keras-applications)
    """

    __name = 'resnet v1 101'
    _module = keras_applications.resnet
    __color_coding = 'rgb'
    __patterns = None

    def __init__(self, include_top=True, weights='imagenet'):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model = keras_applications.resnet.ResNet101(include_top=include_top, weights=weights)
        if include_top:
            self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)
        else:
            self._classifier_model = classifier_model


    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns

class Resnet_v1_152(Keras_App_Model):
    """
    Wrapper for Resnet v1 152 based on keras_applications
    (https://github.com/keras-team/keras-applications)
    """

    __name = 'resnet v1 152'
    _module = keras_applications.resnet
    __color_coding = 'rgb'
    __patterns = None

    def __init__(self, include_top=True, weights='imagenet'):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model = keras_applications.resnet.ResNet152(include_top=include_top, weights=weights)
        if include_top:
            self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)
        else:
            self._classifier_model = classifier_model

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns

class Resnet_v2_50(Keras_App_Model):
    """
    Wrapper for Resnet v2 50 based on keras_applications
    (https://github.com/keras-team/keras-applications)
    """

    __name = 'resnet v2 50'
    _module = keras_applications.resnet_v2
    __color_coding = 'rgb'
    __patterns = None

    def __init__(self, include_top=True, weights='imagenet'):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model = keras_applications.resnet_v2.ResNet50V2(include_top=include_top, weights=weights)
        if include_top:
            self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)
        else:
            self._classifier_model = classifier_model

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns

class Resnet_v2_101(Keras_App_Model):
    """
    Wrapper for Resnet v2 101 based on keras_applications
    (https://github.com/keras-team/keras-applications)
    """

    __name = 'resnet v2 101'
    _module = keras_applications.resnet_v2
    __color_coding = 'rgb'
    __patterns = None

    def __init__(self, include_top=True, weights='imagenet'):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model = keras_applications.resnet_v2.ResNet101V2(include_top=include_top, weights=weights)
        if include_top:
            self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)
        else:
            self._classifier_model = classifier_model
    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns

class Resnet_v2_152(Keras_App_Model):
    """
    Wrapper for Resnet v2 152 based on keras_applications
    (https://github.com/keras-team/keras-applications)
    """

    __name = 'resnet v2 152'
    _module = keras_applications.resnet_v2
    __color_coding = 'rgb'
    __patterns = None

    def __init__(self, include_top=True, weights='imagenet'):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model = keras_applications.resnet_v2.ResNet152V2(include_top=include_top, weights=weights)

        if include_top:
            self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)
        else:
            self._classifier_model = classifier_model

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns

###############################################################################
###############################################################################
###############################################################################

class ResNeXt_50(Keras_App_Model):
    """
    Wrapper for ResNeXt 50 based on keras_applications
    (https://github.com/keras-team/keras-applications)
    """

    __name = 'resnext'
    _module = keras_applications.resnext
    __color_coding = 'rgb'
    __patterns = None

    def __init__(self):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model_with_softmax = keras_applications.resnext.ResNeXt50(include_top=True, weights='imagenet')
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model_with_softmax)

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns

class ResNeXt_101(Keras_App_Model):
    """
        Wrapper for ResNeXt 101 based on keras_applications
        (https://github.com/keras-team/keras-applications)
    """

    __name = 'resnext'
    _module = keras_applications.resnext
    _color_coding = 'rgb'
    __patterns = None

    def __init__(self):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model_with_softmax = keras_applications.resnext.ResNeXt101(include_top=True, weights='imagenet')
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model_with_softmax)

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self._color_coding

    def get_patterns(self):
        return self.__patterns

###############################################################################
###############################################################################
###############################################################################

class Inception_v1(Hub_Model):
    """
        Wrapper for Inception V1 based on Tensorflow Hub
        (https://tfhub.dev/google/imagenet/inception_v1/classification/1)
    """

    __classifier_url = "https://tfhub.dev/google/imagenet/inception_v1/classification/1"  # @param {type:"string"}
    __color_coding = 'rgb'
    __patterns = None

    def __init__(self):
        self._classifier_model = mutils.get_hub_model(self.__classifier_url)

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns

class Inception_v2(Hub_Model):
    """
        Wrapper for Inception V2 based on Tensorflow Hub
        (https://tfhub.dev/google/imagenet/inception_v2/classification/1)
    """

    __classifier_url = "https://tfhub.dev/google/imagenet/inception_v2/classification/1"  # @param {type:"string"}
    __color_coding = 'rgb'
    __patterns = None

    def __init__(self, include_top=True, weights='imagenet'):
        self._classifier_model = mutils.get_hub_model(self.__classifier_url ,include_top=include_top, weights=weights)

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns


class Inception_v3(Keras_App_Model):
    """
        Wrapper for Inception v3 based on Keras Applications
        (https://keras.io/applications/#inceptionv3)
    """

    __name = 'inception_V3'
    _module = keras.applications.inception_v3
    __color_coding = 'rgb'
    __patterns = None

    def __init__(self, include_top=True, weights='imagenet'):
        classifier_model = keras.applications.inception_v3.InceptionV3(
            include_top=include_top,
            weights=weights,
        )
        if include_top:
            self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)
        else:
            self._classifier_model = classifier_model

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns

class Inception_v4:
    """
        Wrapper for Inception V3 based on Kent Sommer's
        Inception V4 implementation
        (https://github.com/kentsommer/keras-inceptionV4/tree/ef1db6f09b6511779c05fab47d374741bc89b5ee)
    """

    color_coding = 'rgb'
    __patterns = None

    IMAGE_SIZE = (299,299)

    def __init__(self, include_top=True, weights='imagenet'):
        classifier_model_with_softmax = inception_v4.create_model(weights=weights, include_top=include_top)
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model_with_softmax)

    def preprocess_input(self, x, **kwargs):
        x = x.convert('RGB')
        #x = np.asarray(x)[:,:,::-1]
        #x = mutils.central_crop(x, 0.875)
        x = x.resize(self.IMAGE_SIZE)
        x = np.array(x)
        x = inception_v4.preprocess_input(x)
        if K.image_data_format() == "channels_first":
            x = np.transpose(x, (2, 0, 1))
            x = x.reshape(-1, 3, 299, 299)
        else:
            x = x.reshape(-1, 299, 299, 3)
        return x

    def predict_wo_softmax(self,x, batch_size=None, verbose=0, steps=None):
        return self._classifier_model.predict(x, batch_size=None, verbose=0, steps=None)

    def predict_with_softmax(self,x, batch_size=None, verbose=0, steps=None):
        preds = self.predict_wo_softmax(x, batch_size=None, verbose=0, steps=None)
        return softmax(preds)

    def decode_predictions(self, preds, top=5, **kwargs):
        return mutils.decode_predictions_inception_v4(preds, top, **kwargs)

    def get_model(self):
        return self.classifier_model

    def get_color_coding(self):
        return self.color_coding

    def get_patterns(self):
        return self.__patterns

class Inception_Resnet_v2(Keras_App_Model):
    """
        Wrapper for Inception ResNet v2 based on Keras Applications
        (https://keras.io/applications/#inceptionresnetv2)
    """

    __name = 'inception_resnet_v2'
    _module = keras.applications.inception_resnet_v2
    __color_coding = 'rgb'
    __patterns = None

    def __init__(self, include_top=True, weights='imagenet'):
        classifier_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=include_top,
            weights=weights,
        )
        if include_top:
            self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)
        else:
            self._classifier_model = classifier_model

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns