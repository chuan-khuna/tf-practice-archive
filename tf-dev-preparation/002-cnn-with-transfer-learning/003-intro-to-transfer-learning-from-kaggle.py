# model
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

# set seed
seed_ = 20200218
tf.random.set_seed(seed_)
np.random.seed(seed_)

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid",
              context="paper",
              font_scale=1.25,
              rc={
                  "figure.figsize": (10.5, 4.5),
                  "figure.dpi": 150,
                  "grid.alpha": 0.1,
                  "grid.color": "#1b262c",
                  "grid.linewidth": 0.5,
                  "font.family": "Operator Mono"
              })
_30k = ["#202f66", "#ff7048", "#7f68d0", "#f3d36e", "#d869ab", "#48ADA9", "#1b262c"]
sns.set_palette(_30k)
import warnings

warnings.filterwarnings('ignore')

# load image configuration
input_shape = (128, 128, 3)

# load pretrained model
# load only pretrained conv
pretrained_vgg = tf.keras.applications.VGG16(
    input_shape=input_shape, include_top=False)
pretrained_vgg.trainable = False

pretrained_vgg.summary()

# Add pretrain to my model
model = tf.keras.models.Sequential()
model.add(pretrained_vgg)
# my dense
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()