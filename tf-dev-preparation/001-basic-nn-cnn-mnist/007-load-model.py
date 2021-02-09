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

# run 006-conv to save model

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = fashion_mnist.load_data()

print(train_imgs.shape, train_labels.shape)

# normalization
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

train_imgs = np.expand_dims(train_imgs, axis=-1)
test_imgs = np.expand_dims(test_imgs, axis=-1)

model = tf.keras.models.load_model("./after-train.h5")

print(model.summary())

print(model.evaluate(train_imgs, train_labels))
