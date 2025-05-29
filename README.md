# Facial Emotion, Age & Gender Detection

## Các yêu cầu cần có:
- Python >= 3.8
- Các thư viện: numpy, pandas, matplotlib, opencv-python, tensorflow, keras, scikit-learn, scikit-image, tqdm, pillow, seaborn, v.v.

## Cách tải dữ liệu

### 1. Dữ liệu nhận diện cảm xúc khuôn mặt (Emotion Recognition)
- Sử dụng tập dữ liệu (https://www.kaggle.com/datasets/msambare/fer2013)
- Sau khi tải về, giải nén dữ liệu. 
- Đường dẫn mặc định của notebook:  
  `C:/Users/PC/Downloads/Detai_TGMT_1/archive/train/`  
  Hãy chỉnh sửa lại biến `train_data_path`, `test_data_path` trong notebook cho phù hợp với đường dẫn mà bạn đang lưu, hoặc đặt dữ liệu đúng vị trí.

### 2. Dữ liệu nhận diện tuổi & giới tính (UTKFace)
- Link: (https://susanqq.github.io/UTKFace/)
- Sau khi tải về, giải nén vào:  
  `C:/Users/PC/Downloads/Detai_TGMT/UTKFace/`
- Có thể đổi lại đường dẫn trong notebook cho phù hợp.


## Hướng dẫn train mô hình

### 1. Nhận diện cảm xúc khuôn mặt

- Chạy notebook:  
  `facial-emotion-detection-using-cnns.ipynb`

- Các bước chính trong notebook:
  - Chuẩn bị dữ liệu: đọc ảnh, xử lý ảnh, gán nhãn.
  - Xây dựng mô hình CNN.
  - Train mô hình với dữ liệu train.
  - Đánh giá trên tập test.

- Lưu ý:
  - Bạn nên kiểm tra lại đường dẫn dữ liệu và chỉnh lại biến tương ứng nếu cần.
  - Có thể chỉnh sửa các siêu tham số (batch size, epochs, learning rate, v.v.) trong phần train.

### 2. Nhận diện tuổi & giới tính

- Chạy notebook:  
  `Age_and_Gender_Detection.ipynb`

- Các bước chính:
  - Đọc tên file, trích xuất nhãn tuổi và giới tính từ tên file.
  - Tiền xử lý ảnh khuôn mặt.
  - Chia train/test/validation.
  - Xây dựng và train mô hình.
  - Dự đoán và đánh giá kết quả.

---

## Inference
### 1. Với mô hình cảm xúc

- Sau khi train xong, có thể sử dụng hàm/đoạn code sau:

```
from tensorflow.keras.models import load_model
import cv2
import numpy as np

IMG_SIZE = 48
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise']

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_gray = img_gray.astype('float32') / 255.0
    img_gray = np.expand_dims(img_gray, axis=(0, -1))
    return img_gray

model = load_model('path_to_saved_model.h5')
image = preprocess_image('path_to_face_image.jpg')
pred = model.predict(image)
emotion = emotion_labels[np.argmax(pred)]
print("Emotion:", emotion)
```

### 2. Với mô hình tuổi & giới tính

- Xem hướng dẫn phần cuối notebook `Age_and_Gender_Detection.ipynb`.
- Thường mô hình sẽ output (age, gender) với gender: 0=Male, 1=Female.

---

## Tham khảo

- [FER2013 Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [Keras Documentation](https://keras.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
