# Gaussian Process Regression Tool 

Deep kernels allow us to train gp regressions without specifically producing a covariance generation function. Train a model on your host machine and export metrics to ./experiments folder as json or use docker containers. The docker container uses the Facebook ax platform for Bayesian Hyperparameter tuning.
To change the underlying dataset replace the dataset.csv file and modify the code to load your X matrix and Y vector in ```gprt/data.py```.


![](images/25_epochs.png)

### Dependancies
Uses Python 3.7+ and GPytorch, dependancies can be found in requirements.txt

### Quickstart
```sh
docker-compose up
```


the hyperparameters, and the default values in ```train_model.py``` that are tuned are:

``` py
# Where to store the results of experiment. Results are stored as pandas.DataFrame objects exported to csv.
# Not needed if used with docker
# Experiments are persisted to mySQl db
RESULTS_ROOT_PATH = "."
# Learning rate of Kernel learning stage
LEARNING_RATE = 0.10
# Rolling integer value to normalize the time series data
ROLLING_VAL = 10
# Number of hidden layers in first layer
N1_LAYERS = 40
# Number of hidden layers in second layer
N2_LAYERS = 20
# Number of Epochs to train
N_EPOCHS = 25
# Size of Kernel embedding grid
GRID_SIZE = 100
```

# Peristing Experiments
Experiments can be loaded from the mySQL databse into pandas or other analytics tools. An example python script is included in ./mySQLTools. ax-service-loop.py uses the ax bayesian optimization framework to run and experiment and persist it for later inspection.
You can run the regression without the ax framework by using