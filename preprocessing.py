import cv2
import numpy as np


def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (128, 128))

    # تحسين التباين (CLAHE) + تقليل الضوضاء
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 3)

    # تحويل لصورة ثنائية (Otsu) لضمان بروز الخطوط
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # تجهيز الصورة للـ CNN
    img_cnn = img_bin.reshape(128, 128, 1) / 255.0

    # حساب الهيستوغرام لاستخدامه كدعم إضافي إذا لزم الأمر
    hist = cv2.calcHist([img_bin], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)

    return img_cnn, hist.flatten()