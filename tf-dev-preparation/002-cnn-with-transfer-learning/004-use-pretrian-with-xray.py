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
    class_mode='binary'
)

print(train_generator.class_indices)

val_datagen = ImageDataGenerator(rescale=1 / 255.0)
val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)

# load pretrained model
vgg_model = tf.keras.applications.VGG16(include_top=False,
                                        input_shape=(target_size[0], target_size[1], 3))
vgg_model.trainable = False

vgg_model.summary()

# create my model
model = tf.keras.models.Sequential()
model.add(vgg_model)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# callbacks
early_stop = tf.keras.callbacks.EarlyStopping(patience=5)
checkpoint = tf.keras.callbacks.ModelCheckpoint("004-best.h5", 
                                                monitor='val_loss', 
                                                save_best_only=True,
                                                save_weights_only=False
                                                )

hist = model.fit_generator(train_generator, epochs=200, callbacks=[early_stop, checkpoint],
                           validation_data=val_generator)

model.save("004-final-trained.h5")


epochs = np.arange(1, len(hist.history['loss']) + 1)
sns.lineplot(epochs, hist.history['loss'], label='loss')
sns.lineplot(epochs, hist.history['val_loss'], label='val loss')
plt.savefig("./004-loss-with-transfer-learning.png", dpi=120)
plt.show()
