import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df_berk = pd.read_csv('Avg_CRU.csv')
df_berk['time'] = pd.to_datetime(df_berk['time'])
df_berk['year'] = df_berk['time'].dt.year
df_berk['month'] = df_berk['time'].dt.month
df_berk['weekofyear'] = df_berk['time'].dt.weekofyear
df_berk['quarter'] = df_berk['time'].dt.quarter
df_berk['dayofyear'] = df_berk['time'].dt.dayofyear
df_berk['dayofweek'] = df_berk['time'].dt.dayofweek
df_berk['hour'] = df_berk['time'].dt.hour
df_berk['minute'] = df_berk['time'].dt.minute
df_berk['second'] = df_berk['time'].dt.second

df_copy = df_berk.copy()
df_copy = df_copy.loc[df_copy['time'] > '1900-01-01', :]
df_copy = df_copy.set_index('time')
df_copy = df_copy.resample('M').mean()
df_copy = df_copy.reset_index()
df_copy.drop('time', axis=1, inplace=True)

train_series = df_copy.loc[(df_copy['year'] >= 1900) & (df_copy['year'] < 2010), :]
test_series = df_copy.loc[(df_copy['year'] >= 2010), :]

additive_decomposition = seasonal_decompose(df_copy['timeseries-tas-monthly-mean'], model='additive', period=12 * 10)

plt.rcParams.update({'figure.figsize': (16, 12)})
additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

autocorrelation_lag1 = df_copy['timeseries-tas-monthly-mean'].autocorr(lag=1)
print("One Month Lag: ", autocorrelation_lag1)

autocorrelation_lag3 = df_copy['timeseries-tas-monthly-mean'].autocorr(lag=3)
print("Three Month Lag: ", autocorrelation_lag3)

autocorrelation_lag6 = df_copy['timeseries-tas-monthly-mean'].autocorr(lag=6)
print("Six Month Lag: ", autocorrelation_lag6)

autocorrelation_lag9 = df_copy['timeseries-tas-monthly-mean'].autocorr(lag=9)
print("Nine Month Lag: ", autocorrelation_lag9)

df_copy['SMA_10'] = df_copy['timeseries-tas-monthly-mean'].rolling(10, min_periods=1).mean()
df_copy['SMA_20'] = df_copy['timeseries-tas-monthly-mean'].rolling(20, min_periods=1).mean()
# Grean = Avg Air Temp, RED = 10 yrs, ORANG colors for the line plot
colors = ['green', 'red', 'orange']
# Line plot
df_copy.plot(color=colors, linewidth=3, figsize=(12, 6))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=['Average air temperature', '10-years SMA', '20-years SMA'], fontsize=14)
plt.title('The yearly average air temperature in city', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Temperature [°C]', fontsize=16)
plt.show()
