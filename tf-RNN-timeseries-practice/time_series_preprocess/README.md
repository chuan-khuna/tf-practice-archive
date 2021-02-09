## วิธีการ preprocess ข้อมูล time series โดยใช้ tensorflow

- อ้างอิงจาก คอร์ส coursera `Sequences, Time Series and Prediction deeplearning.ai`

### ไอเดีย

- ใช้ข้อมูลอดีต window_size ตัว ทำนายอนาคต 1 ตัว
- เช่น window size = 5 `[1, 2, 3, 4, 5]` ทำนายอนาคต `[6]ุ`
