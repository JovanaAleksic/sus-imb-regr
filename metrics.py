'''
Metrics to evaluate performance:
1. RMSE --> root mean squared error
2. MAE --> mean absolute error
'''

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np




def RMSE(y_actual, y_predicted):
	return mean_squared_error(y_actual, y_predicted, squared=False)


def MAE(y_actual, y_predicted):
	# normalized MAE
	return mean_absolute_error(y_actual, y_predicted)


