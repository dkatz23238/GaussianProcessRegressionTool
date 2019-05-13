# Gaussian Process Regression with Deep Kernel.

Deep kernels allow us to train gp regressions without specifically producing a covariance generation function.

![](images/25_epochs.png)

### Dependancies
Uses Python 3.7+ and GPytorch, dependancies can be found in requirements.txt

### Quickstart
``` python gp_singleeval.py ```

the hyperparameters that can be tuned and the default are:

``` py
# Where to store the results of experiment. Results are stored as pandas.DataFrame objects exported to csv.
RESULTS_ROOT_PATH = "."
# Learning rate of Kernel learning stage
LEARNING_RATE = 0.10
# Rolling integer value to normalize the time series data
ROLLING_VAL = 10
# Number of hidden layers in first layer
N1_LAYERS = 100
# Number of hidden layers in second layer
N2_LAYERS = 50
# Number of Epochs to train
N_EPOCHS = 25
# Size of Kernel embedding grid
GRID_SIZE = 100
```