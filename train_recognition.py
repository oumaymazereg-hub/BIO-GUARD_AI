from sklearn.decomposition import PCA
import joblib, dataset, numpy as np, os

if not os.path.exists('models'): os.makedirs('models')

_, X_h, _, Y_p = dataset.load_all_data('data\labelss.csv', 'data\FVC enhanced\cropped_100\crp1_100')
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_h)

joblib.dump(pca, 'models/pca_model.pkl')
np.save('models/features_pca.npy', X_pca)
np.save('models/persons_list.npy', Y_p)
print("✅ PCA model and features saved successfully!")