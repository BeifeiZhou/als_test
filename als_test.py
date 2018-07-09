from implicit.als import AlternatingLeastSquares
import numpy as np
from scipy.sparse import coo_matrix

# This matrix is:
# 1, 0, 1
# 0, 1, 0
# 0, 0, 1
row = np.array([0, 0, 1, 2])
col = np.array([0, 2, 1, 2])
data = np.array([1, 1, 1, 1])


# This matrix is:
# 1, 1, 1
# 1, 0, 1
# 0, 1, 1
# row = np.array([0, 0, 0, 1, 1, 2, 2])
# col = np.array([0, 1, 2, 0, 2, 1, 2])
# data = np.array([1, 1, 1, 1, 1, 1, 1])


# This matrix is:
# 1, 0, 1
# 0, 1, 1
# 1, 0, 0
# row = np.array([0, 0, 1, 1, 2])
# col = np.array([0, 2, 1, 2, 0])
# data = np.array([1, 1, 1, 1, 1])
mat = coo_matrix((data, (row, col)), shape=(3, 3)).tocsr()


# This matrix is:
# 1, 1, 1, 1
# 1, 1, 1, 1
# 1, 1, 0, 1
# 1, 1, 1, 1
# 1, 0, 1, 1
# row = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4])
# col = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 3, 0, 1, 2, 3, 0, 2, 3])
# data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# mat = coo_matrix((data, (row, col)), shape=(5, 4)).tocsr()


print(mat)
model = AlternatingLeastSquares(factors=2, iterations=300, calculate_training_loss=True)
model.fit(mat)
mat_user = model.user_factors
mat_item = model.item_factors
print("User vector: ")
print(mat_user)
print()
print("Item vector: ")
print(mat_item)
print()
print("Predicted R value (i x u): ")
print(mat_item.dot(mat_user.transpose()))
