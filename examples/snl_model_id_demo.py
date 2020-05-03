import numpy as np
import pandas as pd
from models import SparseNonLinearModel

# Simulate a non-linear system of ODEs
f1 = lambda x1, x2: x2 - 2 * x1 + x1**2
f2 = lambda x1, x2: 1 + x1**2

# Generate (X, Y) data samples
X_data = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 1], [3, 3]])
Y_data = np.array([[f1(*x), f2(*x)] for x in X_data]).astype(float)
x_names = ['x_1', 'x_2']  # You can use any names
y_names = ['dx_1/dt', 'dx_2/dt']  # You can use any names

# Use SINDy algorithm to identify non-linear features
model = SparseNonLinearModel(x_names, y_names, poly_order=2)
threshold = 0.1  # sparsity parameter

# Prepare Pandas dataframes with named columns
X = pd.DataFrame(X_data, columns=x_names)
Y = pd.DataFrame(Y_data, columns=y_names)
model.fit(X, Y, threshold=threshold)

# Check fitted model parameters
print(model.coef_.round(4))
print(model.intercept_.round(4))
