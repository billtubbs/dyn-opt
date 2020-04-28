# dyn-opt
Various tools, mostly Python-based, for [system identification][2] and [dynamic optimization][3].

## Contents

### Sparse Identification of Non-linear Dynamics (SINDy)

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
