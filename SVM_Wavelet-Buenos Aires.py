# Replicating the https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6424470
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pywt
from sklearn.datasets import fetch_openml

# Load MNIST data
mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target

# Normalize images to 28x28 pixels
X = X.to_numpy().reshape(-1, 28, 28)

# Apply the CDF 9/7 Wavelet Transform with level adjusted to avoid boundary effects
def apply_wavelet_transform(images, level=1):
    transformed_images = []
    for img in images:
        coeffs = pywt.wavedec2(img, 'bior4.4', level=level)
        # Extract first level coefficients
        LL1 = coeffs[0]  # First level approximation
        LH1, HL1, HH1 = coeffs[1]  # First level details
        
        # Concatenate subbands to form the descriptor
        descriptor = np.hstack((LL1.ravel(), LH1.ravel(), HL1.ravel(), HH1.ravel()))
        transformed_images.append(descriptor)
    return np.array(transformed_images)

# Apply wavelet transform to extract T2 descriptor
wavelet_features = apply_wavelet_transform(X)

# Reduce dimensionality with PCA
pca = PCA(n_components=98)
X_pca = pca.fit_transform(wavelet_features)

# Split data into training and test sets
X_train, X_test = X_pca[:60000], X_pca[60000:]
y_train, y_test = y[:60000], y[60000:]


# Train SVM with RBF kernel
svm = SVC(kernel='rbf', gamma=1/15)
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on MNIST test set: {accuracy * 100:.2f}%") # Accuracy on test set 11.35%
