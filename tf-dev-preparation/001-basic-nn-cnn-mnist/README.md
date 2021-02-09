# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

## Hello World to NN

## Callbacks

### EarlyStopping

หยุดเมื่อ metric ที่ `monitor` ไม่เปลี่ยนแปลงเป็นจำนวน `patience` epoch

กำหนดความเปลี่ยนแปลงน้อยกว่า `min_delta` ได้

```python
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=5)
model.fit(..., callbacks=[early_stop])
```

### custom callback

```python
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= 0.1:
            print("Cancelling training!")
            self.model.stop_training = True

my_callback = MyCallback()
```

## Conv

### CONV layer

- `input_shape=(img_w, img_h, img_ch)` (of 1 image)
- `padding='valid'` ไม่มี padding
- `padding='same'` มี padding ทำให้ output (h, w) เท่ากับ input


### Pooling

a way of compressing image