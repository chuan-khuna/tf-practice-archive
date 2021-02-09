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

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

# model
model = tf.keras.Sequential()
# 1 layer, 1 unit
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

# optimizer
opt = tf.keras.optimizers.SGD(learning_rate=0.01)

model.compile(optimizer=opt, loss='mean_squared_error')

# train
hist = model.fit(xs, ys, epochs=100)

# predict
pred = model.predict([10.0])
print(pred)

epochs = np.arange(1, len(hist.history['loss']) + 1)
sns.lineplot(epochs, hist.history['loss'])
plt.show()
