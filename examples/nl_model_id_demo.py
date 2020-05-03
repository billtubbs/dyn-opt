import numpy as np
import pandas as pd
from dynopt.models.models import NonLinearModel

# Simulate a non-linear system of ODEs
f1 = lambda x1, x2: x2 - 2 * x1 + x1**2
f2 = lambda x1, x2: 1 + x1**2

# Generate (X, Y) data samples
X_data = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 1], [3, 3]])
Y_data = np.array([[f1(*x), f2(*x)] for x in X_data]).astype(float)
x_names = ['x_1', 'x_2']  # You can use any names
y_names = ['dx_1/dt', 'dx_2/dt']  # You can use any names

# Define features to use (more than needed in this example)
x_features = ['1', 'x0', 'x1', 'x0**2', 'x0*x1', 'x1**2']
model = NonLinearModel(x_names, y_names, x_features=x_features)

# Prepare Pandas dataframes with named columns
X = pd.DataFrame(X_data, columns=x_names)
Y = pd.DataFrame(Y_data, columns=y_names)
model.fit(X, Y)

# Check fitted model parameters
print(model.coef_.round(4))
print(model.intercept_.round(4))
