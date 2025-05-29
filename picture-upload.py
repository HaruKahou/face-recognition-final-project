import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model và cascade
model = load_model('C:/Users/Admin/Desktop/TGMT/gender_age_model.h5', compile=False)
face_cascade = cv2.CascadeClassifier('C:/Users/Admin/Desktop/TGMT/haarcascades/haarcascade_frontalface_default.xml')
gender_dict = {0: "Male", 1: "Female"}

# Hàm xử lý ảnh từ file
def predict_from_image(file_path):
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128)) / 255.0
        face_img = face_img.reshape(1, 128, 128, 1)

        predictions = model.predict(face_img, verbose=0)
        gender = gender_dict[round(float(predictions[0][0][0]))]
        age = round(max(0, min(100, float(predictions[1][0][0]))))
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f"{gender}, Age: {age}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Hàm mở file
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        predict_from_image(file_path)

# Gọi lại webcam từ file realtime_detection.py
def start_webcam_detection():
    import subprocess
    subprocess.run(["python", "realtime_detection.py"])

# Giao diện chính
root = tk.Tk()
root.title("Face Recognition Options")
root.geometry("400x200")

label = tk.Label(root, text="Chọn chế độ nhận diện", font=("Arial", 16))
label.pack(pady=20)

btn_upload = tk.Button(root, text="Upload ảnh để dự đoán", command=upload_image, width=30)
btn_upload.pack(pady=10)

btn_realtime = tk.Button(root, text="Dùng webcam (real-time)", command=start_webcam_detection, width=30)
btn_realtime.pack(pady=10)

root.mainloop()
