import itertools
import pandas as pd
from fbp_model import Prophet
from fbp_model.diagnostics import cross_validation
from fbp_model.diagnostics import performance_metrics

df_berk = pd.read_csv('Avg.csv')
df_berk['time'] = pd.to_datetime(df_berk['time'])

df_copy = df_berk[['time', 'temperature']]
df_copy = df_copy.set_index('time')

df_copy = df_copy.loc[df_copy.index > '1900-01-01', :]

df_copy = df_copy.resample('M').mean()

train_series = df_copy.loc[(df_copy.index >= '1900-01-01') & (df_copy.index < '2010-01-01'), 'temperature']
test_series = df_copy.loc[(df_copy.index >= '2010-01-01'), 'temperature']

train_series = train_series.reset_index().rename(columns={'time': 'ds'})
train_series = train_series.rename(columns={'temperature': 'y'})

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []

for params in all_params:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, changepoint_range=0.9,
                **params).fit(train_series)
    df_cv = cross_validation(m, initial='1320 days', period='30 days', horizon='152 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)
