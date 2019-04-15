# Understanding CNNs with Visualizations

The purpose of this repo is to give an understanding 
of CNN's classification decisions using visualizations.
For visualizations the [iNNvestigate toolbox](https://github.com/albermax/innvestigate) is used.

## [Models](https://github.com/infomon/understanding-cnn/tree/master/models)

Pretrained Keras models on imagenet are wrapped up for simple use in experiments.
The following models are available for use:

* [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* [VGG16/19](https://arxiv.org/abs/1409.1556)
* Inception Models [V1](https://arxiv.org/abs/1409.4842),
[V2/3](https://arxiv.org/abs/1512.00567) and
[V4/ResNet V1/2](https://arxiv.org/abs/1602.07261)
* ResNet [V1](https://arxiv.org/abs/1512.03385) and [V2](https://arxiv.org/abs/1603.05027)
* [ResNeXt](https://arxiv.org/abs/1611.05431)

For example, one can load a model and use it as follows:

```python
from models import model_loader

model = model_loader.Inception_v3()
img = Image.open('path/to/img.jpg') # opened with PIL
img_batch = model.preprocess_input(img)
preds = model.predict_with_softmax(img_batch)
print(model.decode_predictions(preds, top=5))
```

## [Notebooks](https://github.com/infomon/understanding-cnn/tree/master/notebooks)
The experiments are executed on Google Colab.
This filter contains multiple notebooks containing the following information:

* [Bottleneck Features](https://github.com/infomon/understanding_cnn/blob/master/notebooks/Bottleneck_features.ipynb): Visualizes bottleneck features. Since bottleneck features are high-dimensional, a dimension reduction algorithm ([UMAP](https://arxiv.org/abs/1802.03426)) is used. 
For each CNN we cut the classifier off. 
Then 50000 examples with 10 different classes ([CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)) are classified and the bottleneck feature maps are stored.
After that UMAP is used to generate visualizations.
* [Compare visualizing methods on MNIST](https://github.com/infomon/understanding_cnn/blob/master/notebooks/Compare_Methods_on_MNIST.ipynb)
Analyze and compare visualizing methods on MNIST images
* [Compare visualizing methods on ImageNet](https://github.com/infomon/understanding_cnn/blob/master/notebooks/Compare_methods_on_ImageNet.ipynb)
Analyze and compare visualizing methods on images contained in [this folder](https://github.com/infomon/understanding_cnn/tree/master/data/images)
Pretrained ImageNet Models are used.
