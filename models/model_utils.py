import keras_applications
import keras
from keras import layers, models
from keras import backend as K
from keras import utils as keras_utils
import numpy as np
from models.alexNet.caffe_classes import class_names
import models.inceptionV4 as inceptionV4
import tensorflow as tf

import tensorflow_hub as hub


def keras_modules_injection(base_fun):
    def wrapper(*args, **kwargs):
        if hasattr(keras_applications, 'get_submodules_from_kwargs'):
            kwargs['backend'] = K
            kwargs['layers'] = layers
            kwargs['models'] = models
            kwargs['utils'] = keras_utils
        return base_fun(*args, **kwargs)

    return wrapper


def decode_predictions(preds, top=5, **kwargs):
    """Decodes the prediction of an ImageNet model.

    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return.
    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.
    # Raises
        ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
    """
    labels_path = keras_utils.get_file('ImageNetLabels.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(imagenet_labels[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results


def decode_predictions_inception_v4(preds, top=5, **kwargs):
    """Decodes the prediction of an ImageNet model.

    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return.
    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.
    # Raises
        ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
    """
    imagenet_labels = eval(open('models/inceptionV4/validation_utils/class_names.txt', 'r').read())
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(imagenet_labels[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results


def decode_predictions_alexNet(preds, top=5, **kwargs):
    """Decodes the prediction of an ImageNet model.

    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return.
    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.
    # Raises
        ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
    """
    imagenet_labels = class_names
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(imagenet_labels[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results


def _get_patterns_info(netname, pattern_type):
    PATTERNS = {
        "vgg16_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
            "url": "https://www.dropbox.com/s/15lip81fzvbgkaa/vgg16_pattern_type_relu_tf_dim_ordering_tf_kernels.npz?dl=1",
            "hash": "8c2abe648e116a93fd5027fab49177b0",
        },
        "vgg19_pattern_type_relu_tf_dim_ordering_tf_kernels.npz": {
            "url": "https://www.dropbox.com/s/nc5empj78rfe9hm/vgg19_pattern_type_relu_tf_dim_ordering_tf_kernels.npz?dl=1",
            "hash": "3258b6c64537156afe75ca7b3be44742",
        },
    }

    if pattern_type is True:
        pattern_type = "relu"

    file_name = ("%s_pattern_type_%s_tf_dim_ordering_tf_kernels.npz" %
                 (netname, pattern_type))

    return {"file_name": file_name,
            "url": PATTERNS[file_name]["url"],
            "hash": PATTERNS[file_name]["hash"]}

def central_crop(image, central_fraction):
    """
    This function comes from Google's ImageNet Preprocessing Script.
    Crop the central region of the image. Remove the outer parts of
    an image but retain the central region of the image along each
    dimension.

    Args:
        image: 3-D array of shape [height, width, depth]
        central_fraction: float (0, 1], fraction of size to crop

    Raises:
        ValueError: if central_crop_fraction is not within (0, 1].

    Returns:
        3-D array

    """

    if central_fraction <= 0.0 or central_fraction > 1.0:
        raise ValueError('central_fraction must be within (0, 1]')
    if central_fraction == 1.0:
        return image

    img_shape = image.shape
    depth = img_shape[2]
    fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
    bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
    bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

    bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
    bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

    image = image[bbox_h_start:bbox_h_start + bbox_h_size, bbox_w_start:bbox_w_start + bbox_w_size]
    return image


def get_hub_model(classifier_url):
    def classifier(x):
        classifier_module = hub.Module(classifier_url)
        return classifier_module(x, signature="image_classification")

    IMAGE_SIZE = get_hub_image_size(classifier_url)
    classifier_layer = layers.Lambda(classifier, input_shape=IMAGE_SIZE + [3])

    classifier_model = keras.models.Sequential([
        classifier_layer
    ])
    classifier_model.summary()

    sess = K.get_session()
    init = tf.global_variables_initializer()

    sess.run(init)

    return classifier_model


def get_hub_image_size(classifier_url):
    return hub.get_expected_image_size(hub.Module(classifier_url))
