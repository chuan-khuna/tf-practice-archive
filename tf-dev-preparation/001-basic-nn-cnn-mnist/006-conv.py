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

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = fashion_mnist.load_data()

print(train_imgs.shape, train_labels.shape)

# normalization
train_imgs = train_imgs/255.0
test_imgs = test_imgs/255.0

train_imgs = np.expand_dims(train_imgs, axis=-1)
test_imgs = np.expand_dims(test_imgs, axis=-1)

print(np.unique(train_labels), len(np.unique(train_labels)))

# Model with CONV
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           padding='same',
                           input_shape=(28, 28, 1),
                           activation='relu'
                           ),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           activation='relu'
                           ),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.save("./before-train.h5", save_format='h5')

hist = model.fit(train_imgs, train_labels,
          epochs=200, callbacks=[early_stop],
          validation_data=(test_imgs, test_labels))

model.save("./after-train.h5", save_format='h5')

epochs = np.arange(1, len(hist.history['loss'])+1)
sns.lineplot(epochs, hist.history['loss'], label='loss')
sns.lineplot(epochs, hist.history['val_loss'], label='val loss')
plt.savefig("./006-loss.png", dpi=120)
plt.show()