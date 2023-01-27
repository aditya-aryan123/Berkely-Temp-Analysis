import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import seaborn as sns
from prophet.plot import plot_yearly
from prophet.plot import add_changepoints_to_plot


color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df_cru = pd.read_csv('../input/Updated_CRU.csv')

df_ft = df_cru[['time', 'timeseries-tas-monthly-mean']].copy()
df_ft['time'] = pd.to_datetime(df_ft['time'])
df_ft.rename(columns={'timeseries-tas-monthly-mean': 'temperature'}, inplace=True)
df_ft.set_index('time', inplace=True)
df_ft = df_ft['temperature'].resample('M').mean()
df_ft = df_ft.reset_index()

train_series = df_ft.loc[(df_ft['time'] >= '1900-01-16') & (df_ft['time'] < '2010-01-16'), :]
test_series = df_ft.loc[(df_ft['time'] >= '2010-01-16'), :]

train_series = train_series.rename(columns={'time': 'ds'})
train_series = train_series.rename(columns={'temperature': 'y'})
test_series = test_series.rename(columns={'time': 'ds'})
test_series = test_series.rename(columns={'temperature': 'y'})

m = Prophet(yearly_seasonality=True, interval_width=0.95, changepoint_range=0.9, changepoint_prior_scale=0.1)
m.fit(train_series)
future = m.make_future_dataframe(freq='M', periods=5*120)
forecast = m.predict(future)
df = test_series.merge(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='left', left_index=True,
                       right_index=True)

ax = df[['y']].plot(figsize=(15, 5))
df['yhat'].plot(ax=ax)
df['yhat_lower'].plot(ax=ax)
df['yhat_upper'].plot(ax=ax)
plt.legend(['Truth Data', 'Truth Data Upper', 'Truth Data Lower', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.show()

