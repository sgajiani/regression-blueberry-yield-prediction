from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np


def compute_eval_metric(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rsq = r2_score(y_test, y_pred)
    return mae, mse, rmse, rsq


