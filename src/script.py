import pandas as pd
import matplotlib.pyplot as plt
from fbp_model import Prophet
import seaborn as sns
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df_berk = pd.read_csv('Avg.csv')
df_berk['time'] = pd.to_datetime(df_berk['time'])

df_copy = df_berk[['time', 'temperature']]
df_copy = df_copy.set_index('time')

df_copy = df_copy.loc[df_copy.index > '1900-01-01', :]

df_copy = df_copy.resample('M').mean()

train_series = df_copy.loc[(df_copy.index >= '1900-01-01') & (df_copy.index < '2010-01-01'), 'temperature']
test_series = df_copy.loc[(df_copy.index >= '2010-01-01'), 'temperature']

fig, ax = plt.subplots(figsize=(20, 7))
train_series.plot(ax=ax, label='Train data')
test_series.plot(ax=ax, label='Test data')
ax.axvline('2010-01-01', color='black', ls='--')
plt.legend(['Train set', 'Test set'], loc='best')
plt.savefig("Original.png", dpi=300)

m = Prophet(seasonality_mode='additive', weekly_seasonality=False, changepoint_prior_scale=0.001,
            seasonality_prior_scale=0.01)
# m.add_seasonality(name='monthly', period=365*12*12, fourier_order=5)
train_series = train_series.reset_index().rename(columns={'time': 'ds'})
train_series = train_series.rename(columns={'temperature': 'y'})
m.fit(train_series)
future = m.make_future_dataframe(freq='M', periods=152)
forecast = m.predict(future)
pd.set_option('display.max_columns', 16)
print(forecast.head())

test_series = train_series.reset_index().rename(columns={'time': 'ds'})
test_series = train_series.rename(columns={'temperature': 'y'})
df_yhat = test_series.merge(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='left', left_index=True,
                            right_index=True)
df_trend = test_series.merge(forecast[['trend', 'trend_lower', 'trend_upper']], how='left', left_index=True,
                             right_index=True)

ax = df_yhat[['y']].plot(figsize=(15, 5))
df_yhat['yhat'].plot(ax=ax, style='.')
df_yhat['yhat_lower'].plot(ax=ax)
df_yhat['yhat_upper'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Truth Data Upper', 'Truth Data Lower', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.savefig("Prediction_yhat.png", dpi=300)

ax = df_trend[['y']].plot(figsize=(15, 5))
df_trend['trend'].plot(ax=ax, style='.')
df_trend['trend_lower'].plot(ax=ax, style='.')
df_trend['trend_upper'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Truth Data Upper', 'Truth Data Lower', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.savefig("Prediction_trend.png", dpi=300)

fig1 = m.plot(forecast)
fig1.savefig('Figure1.png', dpi=300)
