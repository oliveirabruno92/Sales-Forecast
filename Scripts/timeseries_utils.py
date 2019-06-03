from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
import statsmodels.api as sm




def plot_ts(ts, window=12, label='Product quantity', 
            title='Quantity of sales of product P1', ylabel='Quantity of sales'):
    """
    This function plot the time series data and it's rolling statistics.
    param ts: Time series data
    param window: Time window to use on rolling statistics
    param label: Label to put in your time series plot
    param title: Title to put in your time series plot
    param ylabel: Y-axis label to put in your time series plot.
    return: It returns a plt figure.
    """
    ts.plot(figsize=(12, 8), label=label)
    ts.rolling(window=window).mean().plot(label='Rolling Mean')
    ts.rolling(window=window).std().plot(label='Rolling Std')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend()
    
    return plt

def decompose_ts(ts, freq=12, model='additive'):
    """
    This function decomposes a time series in a trend, seasonal and residual components and plots it's components.
    
    param ts: Time series data
    param freq: Frequency for decomposition
    param model: Model for the time series
    returns: It returns it's residual components.
    
    """
    res = sm.tsa.seasonal_decompose(ts.values, freq=freq, model=model)
    
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0,0].plot(ts.index, res.observed)
    ax[0,0].set_xticklabels(labels=ts.index.astype(str), rotation=45)
    ax[0,1].plot(ts.index, res.trend)
    ax[0,1].set_xticklabels(labels=ts.index.astype(str), rotation=45)
    ax[1,0].plot(ts.index, res.seasonal)
    ax[1,0].set_xticklabels(labels=ts.index.astype(str), rotation=45)
    ax[1,1].plot(ts.index, res.observed)
    ax[1,1].set_xticklabels(labels=ts.index.astype(str), rotation=45)
    plt.tight_layout()
    
    return res

def test_stationarity(ts):
    """
    Performs a dickey fuller test to check stationarity of a time series.
    
    param ts: Time series data
    return: Output of the test as a dataframe.
    
    """
    print('Performing Dickey Fuller test..')
    test, pvalue, usedLags, nobs, critical_values, _, _ = adfuller(ts, autolag='AIC')
    output = pd.DataFrame({'TestStatistics': test, 'Pvalue': pvalue, 'usedLags': usedLags, 'Number of obs': nobs}, 
                          index=['Test Statistics', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    for key, value in critical_values.items():
        output.loc['Critical value {}'.format(key)] = value
    
    return output