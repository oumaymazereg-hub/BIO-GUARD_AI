import pandas as pd
import os
import numpy as np
from preprocessing import preprocess_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_all_data(csv_path, img_dir):
    df = pd.read_csv(csv_path, sep=';')
    X_img, X_hist, Y_class, Y_person = [], [], [], []

    # أداة لزيادة البيانات (Augmentation)
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                 height_shift_range=0.1, zoom_range=0.1, fill_mode='nearest')

    for _, row in df.iterrows():
        fname = row['filename']
        # تصحيح الأسماء
        parts = fname.split('_')
        if len(parts[0]) == 1:
            parts[0] = "00" + parts[0]
        elif len(parts[0]) == 2:
            parts[0] = "0" + parts[0]
        fixed_name = "_".join(parts)

        path = os.path.join(img_dir, fixed_name)
        res = preprocess_image(path)

        if res:
            img, hist = res[0], res[1]
            label = row['label'].strip().upper()

            # --- دمج الأصناف ---
            if label in ['ARCH', 'TENTED ARCH', 'TENTED_ARCH']:
                final_label = 'ARCH'
                # مضاعفة بيانات الـ ARCH لتحقيق التوازن (Augmentation)
                # نأخذ الصورة الأصلية + نسختين معدلتين
                X_img.append(img);
                X_hist.append(hist);
                Y_class.append(final_label);
                Y_person.append(row['person'])

                img_reshaped = img.reshape(1, 128, 128, 1)
                for _ in range(2):  # إضافة نسختين إضافيتين لكل صورة Arch
                    aug_img = next(datagen.flow(img_reshaped, batch_size=1))[0]
                    X_img.append(aug_img);
                    X_hist.append(hist);
                    Y_class.append(final_label);
                    Y_person.append(row['person'])
            else:
                X_img.append(img);
                X_hist.append(hist);
                Y_class.append(label);
                Y_person.append(row['person'])

    return np.array(X_img), np.array(X_hist), np.array(Y_class), np.array(Y_person)