import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = np.random.choice([0, 1], size=(20, 20), p=[0.8, 0.2])
mask = data != 0  # 1 where non-zero
sns.heatmap(data, cmap="Greys", cbar=False)
plt.title("Sparsity Mask (White = Non-zero)")
plt.axis('off')
plt.show()
