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
        University of Toronto
        (http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)
    """

    _IMAGE_SIZE = (227,227)
    __color_coding = 'bgr'
    __patterns = None

    def __init__(self):
        def get_alexNet_model_wo_softmax(x):
            filepath = 'models/alexNet/bvlc_alexnet.npy'

            net_data = np.load(open(filepath, "rb"), encoding="latin1").item()

            def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
                '''From https://github.com/ethereon/caffe-tensorflow
                '''
                c_i = input.get_shape()[-1]
                assert c_i % group == 0
                assert c_o % group == 0
                convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

                if group == 1:
                    conv = convolve(input, kernel)
                else:
                    input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
                    kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
                    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                    conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
                return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

            # x = tf.placeholder(tf.float32, (None,) + xdim)

            # conv1
            # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
            k_h = 11;
            k_w = 11;
            c_o = 96;
            s_h = 4;
            s_w = 4
            conv1W = tf.Variable(net_data["conv1"][0])
            conv1b = tf.Variable(net_data["conv1"][1])
            conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
            conv1 = tf.nn.relu(conv1_in)

            # lrn1
            # lrn(2, 2e-05, 0.75, name='norm1')
            radius = 2;
            alpha = 2e-05;
            beta = 0.75;
            bias = 1.0
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

            # maxpool1
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
            k_h = 3;
            k_w = 3;
            s_h = 2;
            s_w = 2;
            padding = 'VALID'
            maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            # conv2
            # conv(5, 5, 256, 1, 1, group=2, name='conv2')
            k_h = 5;
            k_w = 5;
            c_o = 256;
            s_h = 1;
            s_w = 1;
            group = 2
            conv2W = tf.Variable(net_data["conv2"][0])
            conv2b = tf.Variable(net_data["conv2"][1])
            conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv2 = tf.nn.relu(conv2_in)

            # lrn2
            # lrn(2, 2e-05, 0.75, name='norm2')
            radius = 2;
            alpha = 2e-05;
            beta = 0.75;
            bias = 1.0
            lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

            # maxpool2
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            k_h = 3;
            k_w = 3;
            s_h = 2;
            s_w = 2;
            padding = 'VALID'
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            # conv3
            # conv(3, 3, 384, 1, 1, name='conv3')
            k_h = 3;
            k_w = 3;
            c_o = 384;
            s_h = 1;
            s_w = 1;
            group = 1
            conv3W = tf.Variable(net_data["conv3"][0])
            conv3b = tf.Variable(net_data["conv3"][1])
            conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv3 = tf.nn.relu(conv3_in)

            # conv4
            # conv(3, 3, 384, 1, 1, group=2, name='conv4')
            k_h = 3;
            k_w = 3;
            c_o = 384;
            s_h = 1;
            s_w = 1;
            group = 2
            conv4W = tf.Variable(net_data["conv4"][0])
            conv4b = tf.Variable(net_data["conv4"][1])
            conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv4 = tf.nn.relu(conv4_in)

            # conv5
            # conv(3, 3, 256, 1, 1, group=2, name='conv5')
            k_h = 3;
            k_w = 3;
            c_o = 256;
            s_h = 1;
            s_w = 1;
            group = 2
            conv5W = tf.Variable(net_data["conv5"][0])
            conv5b = tf.Variable(net_data["conv5"][1])
            conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv5 = tf.nn.relu(conv5_in)

            # maxpool5
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
            k_h = 3;
            k_w = 3;
            s_h = 2;
            s_w = 2;
            padding = 'VALID'
            maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            # fc6
            # fc(4096, name='fc6')
            fc6W = tf.Variable(net_data["fc6"][0])
            fc6b = tf.Variable(net_data["fc6"][1])
            fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

            # fc7
            # fc(4096, name='fc7')
            fc7W = tf.Variable(net_data["fc7"][0])
            fc7b = tf.Variable(net_data["fc7"][1])
            fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

            # fc8
            # fc(1000, relu=False, name='fc8')
            fc8W = tf.Variable(net_data["fc8"][0])
            fc8b = tf.Variable(net_data["fc8"][1])
            fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

            # prob
            # softmax(name='prob'))
            # prob = tf.nn.softmax(fc8)

            init = tf.initialize_all_variables()
            sess = tf.Session()
            sess.run(init)

            return fc8

        input = layers.Input(shape=list(self._IMAGE_SIZE) + [3])
        network_output = layers.Lambda(get_alexNet_model_wo_softmax)(input)
        self._classifier_model = keras.models.Model(
            inputs=[input],
            outputs=[network_output]
        )

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
    patterns = None

    def __init__(self):
        classifier_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)

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
    patterns = None

    def __init__(self):
        classifier_model = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)

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

    def __init__(self):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model_with_softmax = keras_applications.resnet.ResNet50(include_top=True, weights='imagenet')
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model_with_softmax)

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

    def __init__(self):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model_with_softmax = keras_applications.resnet.ResNet101(include_top=True, weights='imagenet')
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model_with_softmax)

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

    def __init__(self):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model_with_softmax = keras_applications.resnet.ResNet152(include_top=True, weights='imagenet')
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model_with_softmax)

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

    def __init__(self):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model_with_softmax = keras_applications.resnet_v2.ResNet50V2(include_top=True, weights='imagenet')
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model_with_softmax)

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

    def __init__(self):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model_with_softmax = keras_applications.resnet_v2.ResNet101V2(include_top=True, weights='imagenet')
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model_with_softmax)

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

    def __init__(self):
        self._module.decode_predictions = mutils.keras_modules_injection(self._module.decode_predictions)
        self._module.preprocess_input = mutils.keras_modules_injection(self._module.preprocess_input)

        for app in dir(self._module):
            if app[0].isupper():
                setattr(self._module, app, mutils.keras_modules_injection(getattr(self._module, app)))
        setattr(keras_applications, self.__name, self._module)

        classifier_model_with_softmax = keras_applications.resnet_v2.ResNet152V2(include_top=True, weights='imagenet')
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model_with_softmax)

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

    def __init__(self):
        self._classifier_model = mutils.get_hub_model(self.__classifier_url)

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
    patterns = None

    def __init__(self):
        classifier_model = keras.applications.inception_v3.InceptionV3(
            include_top=True,
            weights='imagenet',
        )
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)

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
    patterns = None

    IMAGE_SIZE = (299,299)

    def __init__(self):
        classifier_model_with_softmax = inception_v4.create_model(weights='imagenet', include_top=True)
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
        return self.patterns

class Inception_Resnet_v2(Keras_App_Model):
    """
        Wrapper for Inception ResNet v2 based on Keras Applications
        (https://keras.io/applications/#inceptionresnetv2)
    """

    __name = 'inception_resnet_v2'
    _module = keras.applications.inception_resnet_v2
    __color_coding = 'rgb'
    patterns = None

    def __init__(self):
        classifier_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=True,
            weights='imagenet',
        )
        self._classifier_model = iutils.keras.graph.model_wo_softmax(classifier_model)

    def get_model(self):
        return self._classifier_model

    def get_color_coding(self):
        return self.__color_coding

    def get_patterns(self):
        return self.__patterns