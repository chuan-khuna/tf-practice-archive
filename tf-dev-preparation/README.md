# Tensorflow Dev Exam Preparation

- เตรียมสอบ tensorflow certificate

- จะพยายามใช้ `.py` ไม่ใช้ `notebook`
  - เพื่อให้ชินกับ pycharm ตาม editor ที่ใช้สอบ
  - (สามารถเซฟโมเดลจากที่เทรนจากที่อื่น มาใช้ส่งได้)
- ใช้ editor เป็น `vscode`, `pycharm`

## Version

```txt
python 3.8.5
```

## จาก Bigdata RPG

https://www.youtube.com/watch?v=6L6SpDWG6-8

- Python 3.8 **เท่านั้น**
- tensorflow 2.x
- save model เป็น `.h5`

### 1. NN

- 1 column feature
- 1 output label

### 2. Dataset ง่ายๆ image classification (MNIST, handwriting)

- classification (10 class)

### 3. image classification (advance) - 300x300px RGB

- 2 classification
- augmentation
- image data generator
- transfer learning

> Boyld รู้สึกยาก ได้ 4/5

### 4. NLP

- english ตัดคำไม่ยาก แยกด้วย `spacebar`
- sequential model `LSTM, RNN, GRU, CONV`

### 5. Time Series (sequential)

- pattern, seasonal, noise

## Colab

- student email

```txt
!python --version
Python 3.6.9

import tensorflow as tf
tf.__version__
2.4.0
```

- `runtime` -> `ประเภท runtime` -> GPU

- ช้ากว่าโน้ตบุคอีก CNN 25s vs 70s

## Save Model

- ลงท้ายด้วย `.h5` เป็นการเซฟแบบ `HDF5`

```python
model.save("mymodel.h5")

model = tf.keras.models.load_model("./mymodel.h5")
```

## Callbacks CheatSheet

```python
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint("best.h5",
                                                monitor='val_loss',
                                                save_best_only=True,
                                                save_weights_only=False
                                                )
```
