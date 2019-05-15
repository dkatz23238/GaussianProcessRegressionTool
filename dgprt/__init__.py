import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import os.path
from scipy.io import loadmat
from math import floor
import numpy as np
import pandas as pd
from sklearn import preprocessing

def plot_results(results):
    plt.style.use('ggplot')
    plt.title("GPRegression with GPytorch")
    plt.scatter(results.index.tolist(), results.y_true.values,color="blue", label="true", alpha=0.5)
    plt.plot(results.index, results.predicted.values, label="pred")
    plt.fill_between(results.index.tolist(), results.lower.values, results.upper.values, color="green", label="95%", alpha=0.5)
    plt.legend()
    plt.show()

class DeepNNKernel(torch.nn.Sequential):
    def __init__(self, n_dims, n1_layers, n2_layers):
        super(DeepNNKernel, self).__init__()
        self.add_module('linear1', torch.nn.Linear(n_dims, n1_layers))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(n1_layers, n2_layers))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(n2_layers, 2))


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, grid_size):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)), num_dims=2, grid_size=grid_size)

        self.feature_extractor = feature_extractor

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)

        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main(LEARNING_RATE, ROLLING_VAL, N1_LAYERS, N2_LAYERS, N_EPOCHS, GRID_SIZE):
    # Load the data
    from dgprt.data import X, Y
    Y = (Y - Y.rolling(10).mean()) / Y.rolling(10).std()

    Y = torch.tensor(Y.values[10:]).cpu().float()
    X = torch.tensor(X.values[10:, :]).cpu().float()

    # Use the first 80% of the data for training, and the last 20% for testing.
    train_n = int(floor(0.8*len(X)))
    train_x = X[:train_n, :]
    train_y = Y[:train_n]
    test_x = X[train_n:, :]
    test_y = Y[train_n:]

    n_dims = train_x.size(-1)
    feature_extractor = DeepNNKernel(
        n_dims=n_dims, n1_layers=N1_LAYERS, n2_layers=N2_LAYERS).cpu()
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cpu()
    model = GPRegressionModel(train_x, train_y, likelihood,
                              grid_size=GRID_SIZE, feature_extractor=feature_extractor).cpu()
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=LEARNING_RATE)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def train(N_EPOCHS):
        for i in range(N_EPOCHS):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, N_EPOCHS, loss.item()))
            optimizer.step()

    # See dkl_mnist.ipynb for explanation of this flag
    with gpytorch.settings.use_toeplitz(True):
        train(N_EPOCHS)

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = model((test_x))

    mae = torch.mean(torch.abs(preds.mean - test_y))

    results = pd.DataFrame()
    lower, upper = preds.confidence_region()
    results["y_true"] = test_y.numpy()
    results["predicted"] = preds.mean.numpy()
    results["lower"] = lower.detach().numpy()
    results["upper"] = upper.detach().numpy()

    return results, model

