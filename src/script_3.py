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

m = Prophet(seasonality_mode='additive', daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
m.add_seasonality(name='monthly', period=52568, fourier_order=5)
train_series = train_series.reset_index().rename(columns={'time': 'ds'})
train_series = train_series.rename(columns={'temperature': 'y'})
m.fit(train_series)
future = m.make_future_dataframe(freq='M', periods=152)
forecast = m.predict(future)
pd.set_option('display.max_columns', 16)

test_series = train_series.reset_index().rename(columns={'time': 'ds'})
test_series = train_series.rename(columns={'temperature': 'y'})

df = test_series.merge(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='left', left_index=True,
                       right_index=True)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df['y'])
ax.plot(df['yhat'])
ax.plot(df['yhat_lower'])
ax.plot(df['yhat_upper'])
plt.legend(['Truth Data', 'Truth Data Upper', 'Truth Data Lower', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.savefig("Prediction_yhat_2.png", dpi=300)

fig1 = m.plot(forecast)
fig1.savefig('Figure_1.png', dpi=300)
