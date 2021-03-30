#!/usr/env/python3

from sysidentpy.polynomial_basis import PolynomialNarmax
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.metrics import mean_squared_error
from sysidentpy.utils.generate_data import get_siso_data


# Generate a dataset of a simulated dynamical system
x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
                                                   colored_noise=False,
                                                   sigma=0.001,
                                                   train_percentage=80)

print(type(x_train))

model = PolynomialNarmax(non_degree=2,
                         order_selection=True,
                         n_info_values=10,
                         extended_least_squares=False,
                         ylag=2, xlag=2,
                         info_criteria='aic',
                         estimator='least_squares'
                         )
print(x_valid.shape)
print(y_valid.shape)

model.fit(x_train, y_train)

print()
yhat = model.predict(x_valid, y_valid)
results = pd.DataFrame(model.results(err_precision=8,
                                     dtype='dec'),
                       columns=['Regressors', 'Parameters', 'ERR'])

print(results)

# Regressors     Parameters        ERR
# 0        x1(k-2)     0.9000  0.95556574
# 1         y(k-1)     0.1999  0.04107943
# 2  x1(k-1)y(k-1)     0.1000  0.00335113

print(x_valid.shape, y_valid.shape, yhat.shape)
ee, ex, extras, lam = model.residuals(x_valid, y_valid, yhat)
model.plot_result(y_valid, yhat, ee, ex)
