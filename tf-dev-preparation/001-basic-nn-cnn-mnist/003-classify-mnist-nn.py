# model
import numpy as np
import pandas as pd
import tensorflow as tf

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
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

print(np.unique(train_labels), len(np.unique(train_labels)))

# model
model = tf.keras.Sequential()
# 2-d image to 1-d array
model.add(tf.keras.layers.Flatten())
# hidden layer
model.add(tf.keras.layers.Dense(128, activation='relu'))
# 10-class classification, multi-class classification
# use softmax activation function
# only 1 class output (1-label classification)
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

# sparse categorical crossentropy
# input y label (number) -> pred (one-hot)
model.compile(opt, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

num_epochs = 10
hist = model.fit(train_imgs, train_labels,
                 epochs=num_epochs,
                 validation_data=(test_imgs, test_labels))

# plot
epochs = np.arange(1, num_epochs + 1)
sns.lineplot(epochs, hist.history['loss'], label='loss')
sns.lineplot(epochs, hist.history['val_loss'], label='val loss')
plt.show()
