# Face Recognition Final Project

Dự án này cung cấp các giải pháp nhận diện khuôn mặt, phát hiện cảm xúc, nhận diện tuổi và giới tính trên ảnh cũng như hỗ trợ nhận diện real-time qua webcam. Dưới đây là hướng dẫn chi tiết cách tải dữ liệu, huấn luyện (training) mô hình và chạy inference dự đoán kết quả.

---

## 1. Chuẩn bị môi trường

**Yêu cầu:**
- Python >= 3.8
- Cài đặt các thư viện cần thiết:
  
```bash
pip install -r requirements.txt
```
## 2. Tải dữ liệu

### 2.1. Dữ liệu nhận diện tuổi và giới tính

- **Bộ dữ liệu:** [UTKFace](https://susanqq.github.io/UTKFace/)
- **Cách tải:**
    - Truy cập [link này](https://susanqq.github.io/UTKFace/).
    - Tải file zip về, giải nén ra thư mục, ví dụ: `C:/Users/PC/Downloads/Detai_TGMT/UTKFace`
    - Đảm bảo các file ảnh `.jpg` nằm trong thư mục này.

### 2.2. Dữ liệu nhận diện cảm xúc khuôn mặt

- **Bộ dữ liệu:** [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Cách tải:**
    - Đăng nhập vào Kaggle, tải về bộ dữ liệu FER2013.
    - Giải nén, đặt thư mục train/test về tương ứng, ví dụ:
        - `C:/Users/PC/Downloads/Detai_TGMT_1/archive/train/`
        - `C:/Users/PC/Downloads/Detai_TGMT_1/archive/test/`
    - Cấu trúc thư mục trong `train/` và `test/` sẽ có các folder tương ứng với từng cảm xúc (`angry`, `happy`, ...).

---

## 3. Training mô hình

### 3.1. Training nhận diện tuổi & giới tính

- Mở file notebook [`Age_and_Gender_Detection.ipynb`](./Age_and_Gender_Detection.ipynb).
- Chỉnh sửa lại đường dẫn thư mục chứa ảnh nếu cần thiết.
- Chạy toàn bộ notebook để:
    - Tiền xử lý dữ liệu
    - Tạo dataset
    - Xây dựng và huấn luyện mô hình
    - Lưu lại mô hình (`gender_age_model.h5`)
- *Lưu ý*: Nếu máy yếu, có thể giảm số lượng epoch hoặc batch size.

### 3.2. Training nhận diện cảm xúc

- Mở file [`facial-emotion-detection-using-cnns.ipynb`](./facial-emotion-detection-using-cnns.ipynb).
- Chỉnh sửa lại đường dẫn thư mục chứa dữ liệu train/test cho phù hợp.
- Chạy tuần tự các cell để:
    - Tiền xử lý ảnh (resize, grayscale, cân bằng histogram...)
    - Trích xuất đặc trưng (feature extraction)
    - Xây dựng & huấn luyện mô hình CNN
    - Lưu lại mô hình cảm xúc (`emotion_model.h5`)

---

## 4. Inference - Dự đoán kết quả

### 4.1. Dự đoán tuổi & giới tính từ ảnh hoặc webcam

- Chạy script `picture-upload.py`:
    ```bash
    python picture-upload.py
    ```
- Giao diện sẽ hiện ra cho phép lựa chọn:
    - **Upload ảnh:** Chọn file ảnh bất kỳ, hệ thống sẽ hiển thị dự đoán tuổi và giới tính trên từng khuôn mặt nhận diện được.
    - **Real-time webcam:** Nhấn để bật nhận diện trực tiếp qua webcam (yêu cầu có camera).

### 4.2. Dự đoán cảm xúc

- Sử dụng notebook đã train hoặc xây dựng script inference riêng, nạp mô hình `emotion_model.h5` để dự đoán cảm xúc từ ảnh.

---

## 5. Một số lưu ý

- Các file `.ipynb` nên chạy trên môi trường Jupyter Notebook hoặc Google Colab.
- Đảm bảo các thư mục dữ liệu đúng cấu trúc và đường dẫn trong mã nguồn.
- Nếu gặp lỗi thiếu thư viện, hãy kiểm tra lại các package đã cài đặt.
- Khi chạy trên Windows, chú ý đường dẫn phân tách bằng dấu `\\` hoặc dùng raw string `r"..."`.

---

## 6. Tác giả & Đóng góp

- Dự án thực hiện bởi [HaruKahou](https://github.com/HaruKahou) & team.
- Đóng góp ý kiến, báo lỗi hoặc pull request đều được hoan nghênh!

---

Chúc bạn thành công và có trải nghiệm tốt với dự án nhận diện khuôn mặt đa nhiệm này!