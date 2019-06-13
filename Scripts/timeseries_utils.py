from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

def plot_ts(series, window, ax, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
    This function plot the time series with confidence intervals and anomalies.
    
    param series: Time series data
    param window: Window to calculate rolling statistics
    param ax: Axes to plot 
    param plot_intervals: If to plot confidence intervals or not.
    param plot_anomalies: It to plot anomalies or not.
    return: Figure plot
    
    """
    rolling_mean = series.rolling(window=window).mean()

#     ax.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        ax.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        ax.plot(lower_bond, "r--",)
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            ax.plot(anomalies, "ro", markersize=10)
        
    ax.plot(series[window:], label="Actual values")
    ax.legend(loc=0)
    ax.grid(True)
    ax.set_xticklabels(series.index.astype(str), rotation=45)
    plt.tight_layout()
    
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
#     print('Performing Dickey Fuller test..')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.DataFrame(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'], columns=['Statistical Values'])
    
    for key, value in dftest[4].items():
        dfoutput.loc['Critical value {}'.format(key)] = value
    
    return dfoutput