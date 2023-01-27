import pandas as pd
import numpy as np

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import PolynomialFeatures

df_berk = pd.read_csv('../input/Avg.csv')
df_cru = pd.read_csv('../input/Updated_CRU.csv')

df_berk['time'] = pd.to_datetime(df_berk['time'])
df_cru['time'] = pd.to_datetime(df_cru['time'])

df_berk = df_berk.loc[df_berk['time'] > '1900-01-01', :]
df_cru = df_cru.loc[df_cru['time'] > '1900-01-01', :]

df_berk['year'] = df_berk['time'].dt.year
df_berk["week"] = df_berk['time'].dt.isocalendar().week
df_berk['month'] = df_berk['time'].dt.month
df_berk['weekofyear'] = df_berk['time'].dt.weekofyear
df_berk['quarter'] = df_berk['time'].dt.quarter
df_berk['dayofyear'] = df_berk['time'].dt.dayofyear
df_berk['dayofweek'] = df_berk['time'].dt.dayofweek
df_berk["is_month_start"] = df_berk['time'].dt.is_month_start
df_berk["is_month_end"] = df_berk['time'].dt.is_month_end
df_berk["is_quarter_start"] = df_berk['time'].dt.is_quarter_start
df_berk["is_quarter_end"] = df_berk['time'].dt.is_quarter_end
df_berk["is_year_start"] = df_berk['time'].dt.is_year_start
df_berk["is_year_end"] = df_berk['time'].dt.is_year_end
df_berk["days_in_month"] = df_berk['time'].dt.days_in_month
df_berk["is_leap_year"] = df_berk['time'].dt.is_leap_year
df_berk['is_weekend'] = np.where(df_berk['dayofweek'].isin([5, 6]), 1, 0)

df_cru['year'] = df_cru['time'].dt.year
df_cru["week"] = df_cru['time'].dt.isocalendar().week
df_cru['month'] = df_cru['time'].dt.month
df_cru['weekofyear'] = df_cru['time'].dt.weekofyear
df_cru['quarter'] = df_cru['time'].dt.quarter
df_cru['dayofyear'] = df_cru['time'].dt.dayofyear
df_cru['dayofweek'] = df_cru['time'].dt.dayofweek
df_cru["is_month_start"] = df_cru['time'].dt.is_month_start
df_cru["is_month_end"] = df_cru['time'].dt.is_month_end
df_cru["is_quarter_start"] = df_cru['time'].dt.is_quarter_start
df_cru["is_quarter_end"] = df_cru['time'].dt.is_quarter_end
df_cru["is_year_start"] = df_cru['time'].dt.is_year_start
df_cru["is_year_end"] = df_cru['time'].dt.is_year_end
df_cru["days_in_month"] = df_cru['time'].dt.days_in_month
df_cru["is_leap_year"] = df_cru['time'].dt.is_leap_year
df_cru['is_weekend'] = np.where(df_cru['dayofweek'].isin([5, 6]), 1, 0)

df_berk = df_berk.set_index('time')
df_berk = df_berk.resample('M').mean()
df_berk = df_berk.reset_index()

df_cru = df_cru.set_index('time')
df_cru = df_cru.resample('M').mean()
df_cru = df_cru.reset_index()

df_cru.to_csv('CRU_Updated.csv', index=False)
df_berk.to_csv('BERK_Updated.csv', index=False)


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


def polynomial_transformers():
    return PolynomialFeatures(
        degree=2,
        interaction_only=False,
        include_bias=False
    )


df_berk["sin_week"] = sin_transformer(7).fit_transform(df_berk['week'])
df_berk["sin_month"] = sin_transformer(12).fit_transform(df_berk['month'])
df_berk["sin_quarter"] = sin_transformer(4).fit_transform(df_berk['quarter'])
df_berk["sin_dayofyear"] = sin_transformer(365).fit_transform(df_berk['dayofyear'])
df_berk['sin_day_of_week'] = sin_transformer(7).fit_transform(df_berk['dayofweek'])

df_berk["cos_week"] = cos_transformer(7).fit_transform(df_berk['week'])
df_berk["cos_month"] = cos_transformer(12).fit_transform(df_berk['month'])
df_berk["cos_quarter"] = cos_transformer(4).fit_transform(df_berk['quarter'])
df_berk["cos_dayofyear"] = cos_transformer(365).fit_transform(df_berk['dayofyear'])
df_berk['cos_day_of_week'] = cos_transformer(7).fit_transform(df_berk['dayofweek'])

spline_week = periodic_spline_transformer(7, n_splines=3).fit_transform(df_berk['week'].to_numpy().reshape(-1, 1))
spline_month = periodic_spline_transformer(12, n_splines=6).fit_transform(
    df_berk['month'].to_numpy().reshape(-1, 1))
spline_quarter = periodic_spline_transformer(4, n_splines=2, degree=2).fit_transform(
    df_berk['quarter'].to_numpy().reshape(-1, 1))
spline_dayofyear = periodic_spline_transformer(365, n_splines=182).fit_transform(
    df_berk['dayofyear'].to_numpy().reshape(-1, 1))
spline_day_of_week = periodic_spline_transformer(7, n_splines=3).fit_transform(
    df_berk['dayofweek'].to_numpy().reshape(-1, 1))

for i in range(spline_week.shape[1]):
    df_berk[f"spline_week_{i}"] = spline_week[:, i]

for i in range(spline_month.shape[1]):
    df_berk[f"spline_month_{i}"] = spline_month[:, i]

for i in range(spline_quarter.shape[1]):
    df_berk[f"spline_quarter_{i}"] = spline_quarter[:, i]

for i in range(spline_dayofyear.shape[1]):
    df_berk[f"spline_dayofyear_{i}"] = spline_dayofyear[:, i]

for i in range(spline_day_of_week.shape[1]):
    df_berk[f"spline_day_of_week_{i}"] = spline_day_of_week[:, i]

poly_week = polynomial_transformers().fit_transform(df_berk['week'].to_numpy().reshape(-1, 1))
poly_month = polynomial_transformers().fit_transform(df_berk['month'].to_numpy().reshape(-1, 1))
poly_quarter = polynomial_transformers().fit_transform(df_berk['quarter'].to_numpy().reshape(-1, 1))
poly_dayofyear = polynomial_transformers().fit_transform(df_berk['dayofyear'].to_numpy().reshape(-1, 1))
poly_day_of_week = polynomial_transformers().fit_transform(df_berk['dayofweek'].to_numpy().reshape(-1, 1))


df_cru["sin_week"] = sin_transformer(7).fit_transform(df_cru['week'])
df_cru["sin_month"] = sin_transformer(12).fit_transform(df_cru['month'])
df_cru["sin_quarter"] = sin_transformer(4).fit_transform(df_cru['quarter'])
df_cru["sin_dayofyear"] = sin_transformer(365).fit_transform(df_cru['dayofyear'])
df_cru['sin_day_of_week'] = sin_transformer(7).fit_transform(df_cru['dayofweek'])

df_cru["cos_week"] = cos_transformer(7).fit_transform(df_cru['week'])
df_cru["cos_month"] = cos_transformer(12).fit_transform(df_cru['month'])
df_cru["cos_quarter"] = cos_transformer(4).fit_transform(df_cru['quarter'])
df_cru["cos_dayofyear"] = cos_transformer(365).fit_transform(df_cru['dayofyear'])
df_cru['cos_day_of_week'] = cos_transformer(7).fit_transform(df_cru['dayofweek'])

spline_week = periodic_spline_transformer(7, n_splines=3).fit_transform(df_cru['week'].to_numpy().reshape(-1, 1))
spline_month = periodic_spline_transformer(12, n_splines=6).fit_transform(
    df_cru['month'].to_numpy().reshape(-1, 1))
spline_quarter = periodic_spline_transformer(4, n_splines=2, degree=2).fit_transform(
    df_cru['quarter'].to_numpy().reshape(-1, 1))
spline_dayofyear = periodic_spline_transformer(365, n_splines=182).fit_transform(
    df_cru['dayofyear'].to_numpy().reshape(-1, 1))
spline_day_of_week = periodic_spline_transformer(7, n_splines=3).fit_transform(
    df_cru['dayofweek'].to_numpy().reshape(-1, 1))

for i in range(spline_week.shape[1]):
    df_cru[f"spline_week_{i}"] = spline_week[:, i]

for i in range(spline_month.shape[1]):
    df_cru[f"spline_month_{i}"] = spline_month[:, i]

for i in range(spline_quarter.shape[1]):
    df_cru[f"spline_quarter_{i}"] = spline_quarter[:, i]

for i in range(spline_dayofyear.shape[1]):
    df_cru[f"spline_dayofyear_{i}"] = spline_dayofyear[:, i]

for i in range(spline_day_of_week.shape[1]):
    df_cru[f"spline_day_of_week_{i}"] = spline_day_of_week[:, i]

poly_week = polynomial_transformers().fit_transform(df_cru['week'].to_numpy().reshape(-1, 1))
poly_month = polynomial_transformers().fit_transform(df_cru['month'].to_numpy().reshape(-1, 1))
poly_quarter = polynomial_transformers().fit_transform(df_cru['quarter'].to_numpy().reshape(-1, 1))
poly_dayofyear = polynomial_transformers().fit_transform(df_cru['dayofyear'].to_numpy().reshape(-1, 1))
poly_day_of_week = polynomial_transformers().fit_transform(df_cru['dayofweek'].to_numpy().reshape(-1, 1))

df_berk['SMA_10'] = df_berk['temperature'].rolling(10, min_periods=1).mean()
df_berk['SMA_20'] = df_berk['temperature'].rolling(20, min_periods=1).mean()
df_berk['CMA'] = df_berk['temperature'].expanding().mean()
df_berk['EMA_0.1'] = df_berk['temperature'].ewm(alpha=0.1, adjust=False).mean()
df_berk['EMA_0.3'] = df_berk['temperature'].ewm(alpha=0.3, adjust=False).mean()

df_cru['SMA_10'] = df_cru['timeseries-tas-monthly-mean'].rolling(10, min_periods=1).mean()
df_cru['SMA_20'] = df_cru['timeseries-tas-monthly-mean'].rolling(20, min_periods=1).mean()
df_cru['CMA'] = df_cru['timeseries-tas-monthly-mean'].expanding().mean()
df_cru['EMA_0.1'] = df_cru['timeseries-tas-monthly-mean'].ewm(alpha=0.1, adjust=False).mean()
df_cru['EMA_0.3'] = df_cru['timeseries-tas-monthly-mean'].ewm(alpha=0.3, adjust=False).mean()

df_cru.to_csv('../input/Updated_CRU.csv', index=False)
df_berk.to_csv('../input/BERK.csv', index=False)
