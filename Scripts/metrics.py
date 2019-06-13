from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def rmse(true, predictions): 
    """
    This functions calculate the root mean squared error of a predictions
    
    param true: True values of the data
    param predictions: Predicted values
    return: RMSE
    
    """
    true = np.array(true)
    predictions = np.array(predictions)
    return mean_squared_error(true, predictions) ** 0.5

def mape(true, predictions): 
    """
    This functions calculate the mean absolute percentage error.
    
    param true: True values of the data
    param predictions: Predicted values
    return: MAPE
    
    """
    true = np.array(true)
    predictions = np.array(predictions)    
    return np.mean(np.abs((true - predictions)) / true) * 100

def smape(true, predictions):
    """
    This functions calculate the symmetric mean absolute percentage error
    
    param true: True values of the data
    param predictions: Predicted values
    return: sMAPE
    
    """
    
    true = np.array(true)
    predictions = np.array(predictions)
    
    return np.mean(np.abs(true - predictions) * 2/ (np.abs(true) + np.abs(predictions))) * 100

def metrics(true, predictions):
    """
    This functions calculate several metrics of a regression problem.
    
    param true: True values of the data
    param predictions: Predicted values
    return: Metrics dataframe with all metrics of interest.
    
    """
    metrics = pd.DataFrame(columns=['Metric Value'])
    metrics.loc['MAE'] = mean_absolute_error(true, predictions)
    metrics.loc['RMSE'] = rmse(true, predictions)
    metrics.loc['R2'] = r2_score(true, predictions)
    metrics.loc['MAPE'] = mape(true, predictions)
    metrics.loc['sMAPE'] = smape(true, predictions)
    
    return metrics