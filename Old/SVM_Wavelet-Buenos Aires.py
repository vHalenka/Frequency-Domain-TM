# Replicating the https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6424470
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pywt
from sklearn.datasets import fetch_openml

# Load MNIST data
mnist = fetch_openml('mnist_784')
X = mnist.data.to_numpy().reshape(-1, 28, 28)
y = mnist.target

# Wavelet transform function with level 1
def apply_wavelet_transform(images, level=1):
    transformed_images = []
    for img in images:
        coeffs = pywt.wavedec2(img, 'bior4.4', level=level)
        LL1 = coeffs[0]  # First level approximation
        LH1, HL1, HH1 = coeffs[1]  # First level details
        descriptor_LL1 = LL1.ravel()
        descriptor_T2 = np.hstack((LL1.ravel(), LH1.ravel(), HL1.ravel(), HH1.ravel()))
        transformed_images.append((descriptor_LL1, descriptor_T2))
    return transformed_images

# Apply wavelet transform
wavelet_features = apply_wavelet_transform(X)

# Prepare training and test sets
X_train_LL1 = np.array([feat[0] for feat in wavelet_features[:60000]])
X_train_T2 = np.array([feat[1] for feat in wavelet_features[:60000]])
y_train = y[:60000]

X_test_LL1 = np.array([feat[0] for feat in wavelet_features[60000:]])
X_test_T2 = np.array([feat[1] for feat in wavelet_features[60000:]])
y_test = y[60000:]

# Define SVM classifier
def train_and_evaluate_svm(X_train, y_train, X_test, y_test, gamma_value):
    svm = SVC(kernel='rbf', gamma=gamma_value)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Apply PCA
pca_98 = PCA(n_components=98)
X_train_LL1_PCA98 = pca_98.fit_transform(X_train_LL1)
X_test_LL1_PCA98 = pca_98.transform(X_test_LL1)

X_train_T2_PCA98 = pca_98.fit_transform(X_train_T2)
X_test_T2_PCA98 = pca_98.transform(X_test_T2)

# Run experiments and save results
results = []
experiments = [
    ("LL1", X_train_LL1, X_test_LL1, 15),
    ("T2", X_train_T2, X_test_T2, 15),
    ("LL1T2", np.hstack((X_train_LL1, X_train_T2)), np.hstack((X_test_LL1, X_test_T2)), 21),
    ("LL1 PCA 98", X_train_LL1_PCA98, X_test_LL1_PCA98, 15),
    ("T2 PCA 98", X_train_T2_PCA98, X_test_T2_PCA98, 15)
]

for name, X_train_exp, X_test_exp, gamma in experiments:
    accuracy = train_and_evaluate_svm(X_train_exp, y_train, X_test_exp, y_test, gamma)
    results.append(f"{name}: {accuracy * 100:.2f}%")

# Write results to file
with open("BuenosAiresResults.txt", "w") as file:
    file.write("\n".join(results))

print("Results have been saved to BuenosAiresResults.txt")
