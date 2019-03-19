from keras.datasets import mnist
import keras.backend as K
import os
import cv2

def fetch_mnist_data():
    channels_first = K.image_data_format() == "channels_first"
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if channels_first:
        x_train = x_train.reshape(60000, 1, 28, 28)
        x_test = x_test.reshape(10000, 1, 28, 28)
    else:
        x_train = x_train.reshape(60000, 28, 28, 1)
        x_test = x_test.reshape(10000, 28, 28, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    return x_train, y_train, x_test, y_test

def load_imagenet_test_data():
    pass

def load_from_folder(dirpath, IMAGE_SIZE=(224,224)):
    """
    This function loads images (RGB order) from a given folder

    Args:
        dirpath: path to folder with images
        IMAGE_SIZE: resizing size of image

    Returns: list of images

    """

    valid_images = [".jpg", ".jpeg", ".png", ".tga"]
    if os.path.isdir(dirpath):
        image_list = []
        for f in os.listdir(dirpath):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            img = cv2.imread(os.path.join(dirpath,f))
            img = cv2.resize(img, IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_list.append(img)
        return image_list
    print('Error: Path is no dir')
    return []