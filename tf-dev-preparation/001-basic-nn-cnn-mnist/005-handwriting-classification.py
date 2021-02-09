# model
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

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

# load data
ds, info = tfds.load('mnist', as_supervised=True, with_info=True)

# prepare data
x_train = []
y_train = []
x_test = []
y_test = []
for x, y in ds['train']:
    x_train.append(x.numpy())
    y_train.append(y.numpy())
for x, y in ds['test']:
    x_test.append(x.numpy())
    y_test.append(y.numpy())
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

# normalisation
x_train = x_train / 255.0
x_test = x_test / 255.0

print(info)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(np.unique(y_train))

# model
# simple model -> Dense
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=5)

hist = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test), callbacks=[early_stop])

epochs = np.arange(1, len(hist.history['loss']) + 1)
sns.lineplot(epochs, hist.history['loss'], label='loss')
sns.lineplot(epochs, hist.history['val_loss'], label='val loss')
plt.show()
