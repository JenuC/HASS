from scipy import sparse
import numpy as np
dense = np.random.choice([0, 0, 0, 1], size=(100, 100))
sparse_mat = sparse.csr_matrix(dense)
from matplotlib import pyplot as plt
plt.spy(sparse_mat, markersize=1, color='black')
plt.title(f"Sparsity: {1 - sparse_mat.count_nonzero() / dense.size:.2%}")
plt.axis('off')
plt.show()
#print(f"Density: {sparse_mat.nnz / sparse_mat.size:.2%}")
