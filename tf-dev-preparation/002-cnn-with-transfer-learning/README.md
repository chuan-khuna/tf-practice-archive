# Introduction to TensorFlow w4 : Real Word Image

## Image Generator

https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

- directory ที่ subdirectory คือ label

```python
from keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(rescale=1 / 255.0)
train_gen = data_gen.flow_from_directory()
print(train_gen.class_indices)
```

## With Augmentation in Image generator

```python
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=(1.0 / 255),
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2
)
```

## Transfer Learning

learning from kaggle notebook

- https://www.kaggle.com/digvijayyadav/deep-learning-and-transfer-learning-on-covid-19

### Pretrained - VGG

- https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16

```python
pretrained_model = tf.keras.applications.VGG16(
    include_top=False,
    input_shape=input_shape
)

pretrained_model.trainable = False
```

- `include_top`	whether to include the 3 fully-connected layers at the top of the network.
- set `include_top=False` เอามาแค่ส่วน CNN ไม่เอา FC มา
- set `trainable=False` เพราะไม่ต้องการไปยุ่งกับ weight ที่ train มาแล้ว
