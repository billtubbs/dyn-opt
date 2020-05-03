# dyn-opt
Various Python tools for [system identification][2] and [dynamic optimization][3].

## Model Fitting

The file [models.py](models.py) includes a class of models that provide a convenient
interface for running model estimation and evaluation experiments with data.  They
allow you to generate and fit Scikit-learn estimators to data stored in Pandas 
dataframes.  

Because data in a Pandas dataframe is labelled, the models can be configured to use
specific data while ignoring data that is irrelevant.  This means you can easily 
automate model design, testing and evaluation without having to worry about re-sizing
and matcing input and output data sets to each model.  Instead, you pass all the data
and each model selects only the data it was initialized to use.

The models also allow you to specify additional calculated input features which are
automatically calculated prior to fitting using the Pandas `eval` method which allows
expressions as strings (see the nonlinear model fitting example below).

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

The data is put into Pandas dataframes with appropriate column names:

```
# Construct Pandas dataframes
X = pd.DataFrame(data.data, columns=feature_names)
y = pd.DataFrame(data.target, columns=[target_name])

print(X.head)

      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \
0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   
1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   
2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0   
3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0   
4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0   

   PTRATIO       B  LSTAT  
0     15.3  396.90   4.98  
1     17.8  396.90   9.14  
2     17.8  392.83   4.03  
3     18.7  394.63   2.94  
4     18.7  396.90   5.33  
```

Each data series can now be identified by it's column name and the model is
initialized with the input (x) and output (y) names that you want it to use:

```python
# Selected desired inputs and outputs
x_names = ['LSTAT', 'RM', 'TAX']
y_names = ['MEDV']

# Initialize model
model = Model(x_names, y_names)  # Default is a linear model
print(model)

Model(['LSTAT', 'RM', 'TAX'], ['MEDV'], estimator=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False))
```

The model is then fitted in a similar way to Scikit-Learn models except that
the whole data set can now be passed to the model and the model will only
use the columns it requires.

```python
# Fit model
model.fit(X, y)

# Fit score (R-squared)
print(model.score(X, y))

0.6485147669915273
```


## Sparse Identification of Non-linear Dynamics (SINDy)

This code and the examples are from the book [Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control (1st ed.) by Brunton, S. L., & Kutz, J. N. (2019)][1].

1. [Sparse-Identification-with-SINDy-on-Lorenz-system.ipynb](Sparse-Identification-with-SINDy-on-Lorenz-system.ipynb) 
  A demonstration of SINDy identifying the Lorenz system 
2. [Sparse-Identification-with-SINDy-on-Lorenz-system-with-control.ipynb](Sparse-Identification-with-SINDy-on-Lorenz-system-with-control.ipynb)
  A demonstration of SINDYc identifying the forced Lorenz system
3. [sindy.py](sindy.py) Python module used in above examples.

I also recommend checking out the [official PySindy package][4] developed by Brian de Silva et al. at the University of Washington.

[1]: http://www.databookuw.com
[2]: https://en.wikipedia.org/wiki/System_identification
[3]: https://en.wikipedia.org/wiki/Control_(optimal_control_theory)
[4]: https://github.com/dynamicslab/pysindy
