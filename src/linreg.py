import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split

df_cru = pd.read_csv('../input/Updated_CRU.csv')

df_ft = df_cru[['time', 'lat', 'lon', 'timeseries-tas-monthly-mean', 'year', 'month', 'dayofyear', 'quarter', 'week',
                'weekofyear']].copy()

df_ft['time'] = pd.to_datetime(df_ft['time'])
df_ft.set_index('time', inplace=True)
df_ft = df_ft['timeseries-tas-monthly-mean'].resample('M').mean()
df_ft = df_ft.reset_index()

df_ft['year'] = df_ft['time'].dt.year
df_ft["week"] = df_ft['time'].dt.isocalendar().week
df_ft['month'] = df_ft['time'].dt.month
df_ft['weekofyear'] = df_ft['time'].dt.weekofyear
df_ft['quarter'] = df_ft['time'].dt.quarter
df_ft['dayofyear'] = df_ft['time'].dt.dayofyear
df_ft['dayofweek'] = df_ft['time'].dt.dayofweek
df_ft["is_month_start"] = df_ft['time'].dt.is_month_start
df_ft["is_month_end"] = df_ft['time'].dt.is_month_end
df_ft["is_quarter_start"] = df_ft['time'].dt.is_quarter_start
df_ft["is_quarter_end"] = df_ft['time'].dt.is_quarter_end
df_ft["is_year_start"] = df_ft['time'].dt.is_year_start
df_ft["is_year_end"] = df_ft['time'].dt.is_year_end
df_ft["days_in_month"] = df_ft['time'].dt.days_in_month
df_ft["is_leap_year"] = df_ft['time'].dt.is_leap_year
df_ft['is_weekend'] = np.where(df_ft['dayofweek'].isin([5, 6]), 1, 0)

train_series = df_ft.loc[(df_ft['year'] >= 1900) & (df_ft['year'] < 2010), :]
test_series = df_ft.loc[(df_ft['year'] >= 2010), :]

test_copy = test_series.drop(['timeseries-tas-monthly-mean', 'time'], axis=1)

X = train_series.drop(['timeseries-tas-monthly-mean', 'time'], axis=1)
y = train_series['timeseries-tas-monthly-mean'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

linreg = LinearRegression()
pipe_lr = Pipeline([('scaler', StandardScaler()), ('model', linreg)])
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = metrics.mean_squared_error(y_pred, y_test, squared=False)
r_squared = metrics.r2_score(y_pred, y_test)

test_copy.loc[:, 'prediction'] = pipe_lr.predict(test_copy)
test_copy = test_series.merge(test_copy[['prediction']], how='left', left_index=True, right_index=True)
plt.figure(figsize=(10, 8))
ax1 = sns.kdeplot(test_copy['prediction'], color="r", label="Actual Value")
sns.kdeplot(test_copy['timeseries-tas-monthly-mean'], color="b", label="Fitted Values", ax=ax1)
plt.show()

print(f"Model: {linreg}, Mean squared error: {mse}, Root mean squared error: {rmse}, R-Squared: {r_squared}")

lasso = Lasso()
pipe_lasso = Pipeline([('scaler', StandardScaler()), ('model', lasso)])
pipe_lasso.fit(X_train, y_train)
y_pred = pipe_lasso.predict(X_test)
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = metrics.mean_squared_error(y_pred, y_test, squared=False)
r_squared = metrics.r2_score(y_pred, y_test)

print(f"Model: {lasso}, Mean squared error: {mse}, Root mean squared error: {rmse}, R-Squared: {r_squared}")

ridge = Ridge()
pipe_ridge = Pipeline([('scaler', StandardScaler()), ('model', ridge)])
pipe_ridge.fit(X_train, y_train)
y_pred = pipe_ridge.predict(X_test)
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = metrics.mean_squared_error(y_pred, y_test, squared=False)
r_squared = metrics.r2_score(y_pred, y_test)

print(f"Model: {ridge}, Mean squared error: {mse}, Root mean squared error: {rmse}, R-Squared: {r_squared}")

elastic_net = ElasticNet()
pipe_elastic_net = Pipeline([('scaler', StandardScaler()), ('model', elastic_net)])
pipe_elastic_net.fit(X_train, y_train)
y_pred = pipe_elastic_net.predict(X_test)
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = metrics.mean_squared_error(y_pred, y_test, squared=False)
r_squared = metrics.r2_score(y_pred, y_test)

print(f"Model: {elastic_net}, Mean squared error: {mse}, Root mean squared error: {rmse}, R-Squared: {r_squared}")
