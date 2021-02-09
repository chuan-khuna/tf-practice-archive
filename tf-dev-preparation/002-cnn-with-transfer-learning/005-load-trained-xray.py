# model
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

# set seed
seed_ = 20200218
tf.random.set_seed(seed_)
np.random.seed(seed_)

# report
from sklearn.metrics import confusion_matrix, classification_report

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

# prepare data
# image config
target_size = (128, 128)
batch_size = 32

# datagen with augmentation
train_path = "./chest_xray/train/"
val_path = "./chest_xray/test/"

train_datagen = ImageDataGenerator(rescale=1 / 255.0, shear_range=0.2, zoom_range=0.2)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)
val_datagen = ImageDataGenerator(rescale=1 / 255.0)
val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)


train_true = train_generator.classes
val_true = val_generator.classes

print(train_true)

print(train_generator.class_indices)

final_model = tf.keras.models.load_model("./004-final-trained.h5")
best_model = tf.keras.models.load_model("./004-best.h5")

final_model.summary()
best_model.summary()

final_pred_train = final_model.predict(train_generator) > 0.5
final_pred_val = final_model.predict(val_generator) > 0.5

best_pred_train = best_model.predict(train_generator) > 0.5
best_pred_val = best_model.predict(val_generator) > 0.5

print("train\n")
print(classification_report(train_true, final_pred_train))
print("val\n")
print(classification_report(val_true, final_pred_val))

print("\n\n")

print("train\n")
print(classification_report(train_true, best_pred_train))
print("val\n")
print(classification_report(val_true, best_pred_val))

