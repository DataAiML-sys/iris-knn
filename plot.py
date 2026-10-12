import matplotlib.pyplot as plt

# Results from the C++ KNN program
K = [1, 3, 5, 7, 10, 15]
euclidean = [96.00, 95.33, 96.67, 97.33, 97.33, 96.67]
manhattan = [94.00, 95.33, 94.67, 96.00, 95.33, 96.00]
cosine    = [81.33, 81.33, 78.00, 81.33, 80.00, 80.67]

plt.figure(figsize=(9, 5))

plt.plot(K, euclidean, marker='o', label='Euclidean', linewidth=2)
plt.plot(K, manhattan, marker='s', label='Manhattan', linewidth=2)
plt.plot(K, cosine,    marker='^', label='Cosine',    linewidth=2)

plt.xlabel('K (number of neighbors)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('KNN accuracy on Iris dataset (5-fold cross-validation)', fontsize=13)

plt.xticks(K)
plt.ylim(70, 100)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('plot.png', dpi=150)
print('Saved plot.png')