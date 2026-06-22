from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import dataset, classification_model, os, joblib

# تحميل البيانات المدمجة والموزونة
X_i, _, Y_c, _ = dataset.load_all_data('data/labelss.csv', 'data/FVC enhanced/cropped_100/crp1_100')
le = LabelEncoder()
Y_c_enc = le.fit_transform(Y_c)

if not os.path.exists('models'): os.makedirs('models')
joblib.dump(le, 'models/label_encoder.pkl')  # حفظ المترجم

# Cross Validation Folder = 5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_i, Y_c_enc)):
    print(f"🚀 Training Fold {fold + 1}...")
    model = classification_model.build_classifier(len(le.classes_))

    model.fit(X_i[train_idx], Y_c_enc[train_idx],
              validation_data=(X_i[val_idx], Y_c_enc[val_idx]),
              epochs=40, batch_size=32, verbose=1)

    model.save(f'models/class_model_f{fold}.h5')