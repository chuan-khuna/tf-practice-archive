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

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(rescale=1/255.0)

train_gen = data_gen.flow_from_directory("./chest_xray/train/",
                                         target_size=(256, 256),
                                         batch_size=64)