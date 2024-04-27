import numpy as np
from sklearn.manifold import TSNE
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the MNIST dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

print(type(X))
print(X.shape)
print(type(y))
print(y.shape)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_reduced = tsne.fit_transform(X)
print(type(X_reduced))
print(X_reduced.shape)

# # Plot the reduced data
# x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
# y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1


plt.figure(figsize=(8, 8))
for i in range(10):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], label=str(i))

plt.legend()
plt.xticks([])
plt.yticks([])
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title("t-SNE visualization of MNIST dataset") 
plt.show()
# Save the figure
plt.savefig('mnist_tsne.png')