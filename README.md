# dyn-opt
Various Python tools for non-linear function approximation, [system identification][2] 
and [dynamic optimization][3].

## Model Fitting

The file [models.py](models.py) contains a class of models that provide a convenient
interface for running model estimation and evaluation experiments with data.  They
allow you to generate and fit Scikit-learn estimators to data stored in Pandas 
dataframes.  

Because data in a Pandas dataframe is labelled, the models can be configured to use
specific data while ignoring data that is not relevant.  This means you can easily 
automate model design, testing and evaluation with different inputs and outputs,
without having to worry about re-sizing and matcing the data sets to each model.
Instead, you can pass all the data to each model and it will only use the fields
it was intended for.

The models also allow you to specify additional calculated input features which are
automatically calculated prior to model-fitting using the Pandas `eval` method.  
This allows you to define non-linear features as expressions = (see the nonlinear 
model fitting example below).


## Example 1 - Linear regression on a subset of features

For this example we download the Boston housing price dataset:

```python
from sklearn.datasets import load_boston

# Load dataset
data = load_boston()

# Get data names
feature_names = data.feature_names.tolist()
target_name = 'MEDV'  # Median value of homes in $1000s
```

First, put the data into Pandas dataframes with appropriate column names:

```python
X = pd.DataFrame(data.data, columns=feature_names)
y = pd.DataFrame(data.target, columns=[target_name])

print(X.head)

#       CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \
# 0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   
# 1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   
# 2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0   
# 3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0   
# 4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0   
# 
#    PTRATIO       B  LSTAT  
# 0     15.3  396.90   4.98  
# 1     17.8  396.90   9.14  
# 2     17.8  392.83   4.03  
# 3     18.7  394.63   2.94  
# 4     18.7  396.90   5.33  
```

The the input (x) and output (y) data that you want to use can now be identified 
by their column names when initializing the model:

```python
# Select desired inputs and outputs
x_names = ['LSTAT', 'RM', 'TAX']
y_names = ['MEDV']

# Initialize model
model = Model(x_names, y_names)  # by default uses a linear estimator
print(model)

# Model(['LSTAT', 'RM', 'TAX'], ['MEDV'], estimator=LinearRegression(copy_X=True, 
# fit_intercept=True, n_jobs=None, normalize=False))
```

The model is then fitted in a similar way to Scikit-Learn models except that
the whole data set can now be passed to the model and it will only use the 
data columns it requires.

```python
# Fit model
model.fit(X, y)

# Fit score (R-squared)
print(model.score(X, y))

# 0.6485147669915273
```

Likewise, when predicting with the fitted model, only the relevant input data
is used by the model.  The predicted output is a dataframe:

```python
print(model.predict(X.head()))

#         MEDV
# 0  29.012111
# 1  26.263785
# 2  33.059827
# 3  32.819835
# 4  32.273937
```

Single-point prediction using dictionaries is also supported:

```python
x = {'LSTAT': 4.98, 'RM': 6.575, 'TAX': 296}
print(model.predict(x))

# {'MEDV': 29.01211141973685}
```

## Example 2 - Non-Linear Model Estimation

The `NonLinearModel` class in [models.py](models.py) allows you to specify features
as calculated expressions of the input data.

The next two examples demonstrate how to use this feature to identify the non-linear
dynamics of a simple pendulum from data.

First we generate data by simulating a pendulum with two ordinary differential 
equations describing its motion:

```python
import numpy as np

def pendulum_dydt(t, y):
    dydt = np.empty_like(y)
    dydt[0] = y[1]
    dydt[1] = -y[1] - 5*np.sin(y[0])
    return dydt
```

Integrate dy/dt over time:

```python
from scipy.integrate import odeint

t = np.arange(0, 10, 0.1)
y0 = [np.pi+0.1, 0]  # Initial condition
y = odeint(pendulum_dydt, y0, t, tfirst=True)
assert y.shape == (100, 2)
```

The plot below shows how the pendulum states vary over time.

<IMG SRC='images/pendulum-time-plot.png' WIDTH=250>

Next, compute the derivatives that we want to predict:

```python
# Calculate dydt values
dydt = pendulum_dydt(t, y)
assert dydt.shape == (100, 2)
```

Initialize a non-linear estimation model with appropriate inputs and outputs:

```python
from dynopt.models.models import NonLinearModel

# Labels for inputs and outputs
x_names = ['theta', 'theta_dot']
y_names = ['theta_dot', 'theta_ddot']

# Choose input features including non-linear terms
x_features = ['x1', 'sin(x0)']

# Initialize model
model = NonLinearModel(x_names, y_names, x_features=x_features)
```

Prepare pandas dataframes with the same names containing the input and 
output data:

```python
y = pd.DataFrame(dydt, columns=y_names)
X = pd.DataFrame(y, columns=x_names)
print(y.head())

#       theta  theta_dot
# 0  0.100000   0.000000
# 1  0.097595  -0.047109
# 2  0.090802  -0.087513
# 3  0.080359  -0.119943
# 4  0.067106  -0.143621

print(dydt.head())

#    theta_dot  theta_ddot
# 0   0.000000   -0.499167
# 1  -0.047109   -0.440093
# 2  -0.087513   -0.365874
# 3  -0.119943   -0.281420
# 4  -0.143621   -0.191657
```

Fit the model to the data and display the coefficients.

```python
model.fit(y, dydt)
print(model.coef_.round(5))

#      x1  sin(x0)
# y0  1.0     -0.0
# y1 -1.0     -5.0
```

Note that the estimated coefficients are very close to the coefficients in the 
original system equations above.


## Example 3 - Sparse Identification of Non-linear Dynamics

The Sparse Identification of Non-linear Dynamics algorithm (SINDy) is a numerical 
technique that automatically identifies non-linear relationships when the governing 
equations of the system are sparse (i.e. when they have a few dominant terms).  When 
this is the case, the SINDy algorithm finds a sparse approximation of the true 
dynamics.  

Here, we use same data as in the previous example but this time we assume that we don't 
exactly know the terms in the underlying equations.  Instead, we specify a polynomial
model order that we think will capture the true dynamics (2nd order here) as well as
any additional non-linear terms we think may exist (e.g. sine, cosine functions).

The `SparseNonLinearModel` class constructs a data library for all possible polynomial
terms (x0, x1, x0*x1, x0**2, x1**2, ... etc.) as well as any additional terms specified,
and then uses an iterative least-squares procedure to recursively eliminate terms that 
are not significant.  

A threshold parameter determines how many terms are eliminated:

```python
from dynopt.models.models import SparseNonLinearModel

# Choose additional non-linear features for model identification
custom_features = ['sin(x0)', 'cos(x0)', 'sin(x0)**2', 
                   'cos(x0)**2', 'sin(x0)*cos(x0)']

# Initialize SINDy model
model = SparseNonLinearModel(x_names, y_names, 
                             custom_features=custom_features,
                             poly_order=2)
```

Fit model (implements SINDy sparsification procedure):

```python
threshold = 0.2  # Sparsity parameter
model.fit(y, dydt, threshold=threshold)
print(model.n_params)
# 3
```

Display fitted model coefficients:

```python
print(model.coef_)

#      x1  sin(x0)
# y0  1.0      0.0
# y1 -1.0     -5.0
```

Again, if you look at the original governing equations above, you can see that it 
has identified the correct terms as well as the correct coefficients exactly.

The SINDy code in this package is based on the methods and code provided in the book 
[Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control (1st ed.) by Brunton, S. L., & Kutz, J. N. (2019)][1].

The following notebooks show how replicate two of the examples in the book using the code in this package:

1. [Sparse-Identification-with-SINDy-on-Lorenz-system.ipynb](Sparse-Identification-with-SINDy-on-Lorenz-system.ipynb) 
  A demonstration of SINDy identifying the Lorenz system 
2. [Sparse-Identification-with-SINDy-on-Lorenz-system-with-control.ipynb](Sparse-Identification-with-SINDy-on-Lorenz-system-with-control.ipynb)
  A demonstration of SINDYc identifying the forced Lorenz system
3. [sindy.py](sindy.py) Python module used in above examples.

There is also an [official PySindy package][4] developed by Brian de Silva et al. at the University of Washington for implementing SINDy which contains some additional features.

[1]: http://www.databookuw.com
[2]: https://en.wikipedia.org/wiki/System_identification
[3]: https://en.wikipedia.org/wiki/Control_(optimal_control_theory)
[4]: https://github.com/dynamicslab/pysindy
