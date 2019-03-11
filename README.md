# Understanding CNNs with Visualizations

The purpose of this repo is to give an understanding 
of CNN's classification decisions using visualizations.
For visualizations the [iNNvestigate toolbox](https://github.com/albermax/innvestigate) is used.

## [Models](https://github.com/infomon/understanding-cnn/tree/master/models)

Pretrained models on imagenet are wrapped up for simple use in experiments.
The following models are available for use:

* [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* [VGG16/19] (https://arxiv.org/abs/1409.1556)
* Inception Models [V1](https://arxiv.org/abs/1409.4842),
[V2/3](https://arxiv.org/abs/1512.00567) and
[V4/ResNet V1/2](https://arxiv.org/abs/1602.07261)
* ResNet [V1](https://arxiv.org/abs/1512.03385) and [V2](https://arxiv.org/abs/1603.05027)
* [ResNeXt](https://arxiv.org/abs/1611.05431)

For example, one can load a model and use the as follows:

```python
from models import model_loader

model = model_loader.Inception_v3()
img = Image.open('path/to/img.jpg') # opened with PIL
img_batch = model.preprocess_input(img)
preds = model.predict_with_softmax(img_batch)
print(model.decode_predictions(preds, top=5))
```
