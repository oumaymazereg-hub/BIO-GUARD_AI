import numpy as np

def chi_square_distance(H1, H2):
    return 0.5 * np.sum(((H1 - H2)**2) / (H1 + H2 + 1e-10))

def match_person(test_pca, train_pca_list, train_persons_list):
    if len(train_pca_list) == 0: return -1
    dists = [chi_square_distance(test_pca, dp) for dp in train_pca_list]
    return train_persons_list[np.argmin(dists)]