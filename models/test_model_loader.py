from models import model_loader
import keras.backend as backend
import numpy as np
import six
import PIL.Image as Image
import pytest

nets = [
    # NAME                  MODEL LOADER
    ["AlexNet",             model_loader.AlexNet],
    ["VGG16",               model_loader.VGG16],
    ["VGG19",               model_loader.VGG19],
    ["Inception_v1",        model_loader.Inception_v1],
    ["Inception_v2",        model_loader.Inception_v2],
    ["Inception_v3",        model_loader.Inception_v3],
    ["Inception_v4",        model_loader.Inception_v4],
    ["Inception_Resnet_v2", model_loader.Inception_Resnet_v2],
    ["Resnet_v1_50",        model_loader.Resnet_v1_50],
    ["Resnet_v1_101",       model_loader.Resnet_v1_101],
    ["Resnet_v1_152",       model_loader.Resnet_v1_152],
    ["Resnet_v2_50",        model_loader.Resnet_v2_50],
    ["Resnet_v2_101",       model_loader.Resnet_v2_101],
    ["Resnet_v2_152",       model_loader.Resnet_v2_152],
    ["ResNeXt_50",          model_loader.ResNeXt_50],
    ["ResNeXt_101",         model_loader.Resnet_v1_101]
]

def keras_test(func):
    """Function wrapper to clean up after TensorFlow tests.
    # Arguments
        func: test function to clean up after.
    # Returns
        A function wrapping the input function.
    """
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        if backend.backend() == 'tensorflow' or backend.backend() == 'cntk':
            backend.clear_session()
        return output
    return wrapper

@keras_test
def _test_application_basic(loader, img, name):
    net = loader()
    img_batch = net.preprocess_input(img)
    assert isinstance(img_batch, np.ndarray)
    assert img_batch.ndim == 4

    preds = net.predict_with_softmax(img_batch)
    assert len(preds[0]) == 1000 or len(preds[0]) == 1001
    assert 0.9999 <= np.sum(preds) <= 1.0001

    top5 = net.decode_predictions(preds, top=5)
    assert len(top5[0]) == 5

    # Test correct label is in top 5 (weak correctness test).
    for top in top5:
        if 'elephant' in top[0][0].lower():
            return
    print("Warning: Image detected no elephant in image!")


if __name__ == "__main__":
    img = Image.open('images/elephant.jpg')
    for [name,loader] in nets:
        _test_application_basic(loader, img, name)