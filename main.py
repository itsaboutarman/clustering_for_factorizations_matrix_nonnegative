import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment


def get_clustering_accuracy(y_true, y_pred):
    """
    Since clustering labels have no specific order (permutation invariance),
    this function finds the best match between cluster labels and true labels
    using the Hungarian algorithm, then calculates accuracy.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Find the best assignment
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    return cm[row_ind, col_ind].sum() / np.sum(cm)


def plot_gallery(title, images, n_col=5, n_row=2):
    """Helper function to plot images (basis patterns or centroids)"""
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(28, 28), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


# 1. Load MNIST Dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

# Normalize data (NMF requires non-negative data)
# Pixel values are 0-255, scaling them to 0-1
X = X / 255.0

print(f"Data Matrix (X) shape: {X.shape}")

# 2. Run NMF Algorithm
# We use 10 components since there are 10 digits (0-9)
n_components = 10
print(f"Running NMF with {n_components} components on the full dataset...")

nmf = NMF(n_components=n_components, init='random',
          random_state=42, max_iter=500)

# W is the membership matrix (referred to as F in the project description)
W = nmf.fit_transform(X)

# H is the basis matrix (referred to as G in the project description)
# The rows of H contain the basis images
H = nmf.components_

print("NMF execution completed.")

# 3. Visualize the columns of Matrix G (Basis Images)
plot_gallery("NMF Basis Images (Columns of G)", H)
plt.show()

# 4. Clustering using Membership Matrix F (here W)
# Each data point belongs to the cluster corresponding to the highest weight in W
y_pred_nmf = np.argmax(W, axis=1)

# Calculate NMF clustering accuracy
acc_nmf = get_clustering_accuracy(y, y_pred_nmf)
print(f"\nNMF Clustering Accuracy: {acc_nmf:.4f}")

# 5. Compare with K-means
print("\nRunning K-means on the original data...")
kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
y_pred_kmeans = kmeans.fit_predict(X)

# Calculate K-means accuracy
acc_kmeans = get_clustering_accuracy(y, y_pred_kmeans)
print(f"K-means Clustering Accuracy: {acc_kmeans:.4f}")

# 6. Visualize K-means Centroids
plot_gallery("K-means Centroids", kmeans.cluster_centers_)
plt.show()

# Final Analysis
print("-" * 30)
print("Analysis of Results:")
if acc_nmf > acc_kmeans:
    print("In this run, NMF performed better than K-means.")
else:
    print("In this run, K-means performed better (or close) to NMF.")

print("Qualitative Difference:")
print("- NMF basis images typically show 'parts' of digits (strokes, loops).")
print("- K-means centroids show the 'average' of whole digits.")



def plot_gallery(title, images, n_col=5, n_row=5, cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images[:n_col * n_row]):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(28, 28), cmap=cmap,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    plt.savefig("mnist_samples.png") # این فایل ذخیره میشه
    plt.show()

# فراخوانی تابع (جایی که دیتا رو لود کردی)
# X همان دیتای اصلی MNIST است
plot_gallery("نمونه‌هایی از دیتاست MNIST", X[:25])
