

def timeseries_cv(model, n_splits=5):
    """
    This functions performs a cross validation on time series data with RMSE as scoring function.
    
    param model: Estimator
    param n_splits: splits to do on training data.
    return: Scores from cross validation.

    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = np.sqrt(-cross_val_score(estimator=model, X=X, y=y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1))
    
    return scores