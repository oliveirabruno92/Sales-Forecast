from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import numpy as np
from metrics import rmse


def ts_train_test_split(data, target='qty_order', split_date='2015-09-14'):
    """
    This functions performs a train and test split on time series data.
    
    param data: Time series dataframe
    param target: Target to forecast
    param split_date: Date to split the dataset
    returns: Splitted dataset.
    
    """
    X = data.drop(target, axis=1)
    y = data[target]
    
    X_train, X_test = X[:split_date], X[split_date:]
    y_train, y_test = y[:split_date], y[split_date:]
    
    return X_train, X_test, y_train, y_test


def timeseries_cv(model, data, n_splits=5):
    """
    This functions performs a cross validation on time series data with RMSE as scoring function.
    
    param model: Estimator
    param n_splits: splits to do on training data.
    return: Scores from cross validation.

    """
    X_train, X_test, y_train, y_test = ts_train_test_split(data)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = np.sqrt(-cross_val_score(estimator=model, X=X_train.values, y=y_train.values, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1))
    
    return scores


def prediction_pipeline(model, X_train, X_test, y_train, y_test):
    """
    This function performs a pipeline for prediction and score on test set.
    
    param model: Estimator
    param X_train: Train dataframe without target.
    param X_test: Test dataframe without target.
    param y_train: Train target.
    param y_test: Test target.
    return: Predictions and score 
    
    """
    model.fit(X_train, y_train)
    predict_train = model.predict(X_train)
    predict_test = model.predict(X_test)
    score_train = rmse(y_train, predict_train)
    score_test = rmse(y_test, predict_test)
    
    return predict_train, predict_test, score_train, score_test