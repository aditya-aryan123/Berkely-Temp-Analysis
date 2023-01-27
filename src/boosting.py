import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgm
from sklearn.model_selection import train_test_split


df_cru = pd.read_csv('../input/CRU_Updated.csv')
df_cru.drop('time', axis=1, inplace=True)

train_series = df_cru.loc[(df_cru['year'] >= 1900) & (df_cru['year'] < 2010), :]
test_series = df_cru.loc[(df_cru['year'] >= 2010), :]

test_copy = test_series.drop('timeseries-tas-monthly-mean', axis=1)

X = train_series.drop('timeseries-tas-monthly-mean', axis=1)
y = train_series['timeseries-tas-monthly-mean'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

xgb_r = xgb.XGBRegressor()
pipe_xgb_r = Pipeline([('scaler', StandardScaler()), ('model', xgb_r)])
pipe_xgb_r.fit(X_train, y_train)
y_pred = pipe_xgb_r.predict(X_test)
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = metrics.mean_squared_error(y_pred, y_test, squared=False)
r_squared = metrics.r2_score(y_pred, y_test)

print(f"Model: {xgb_r}, Mean squared error: {mse}, Root mean squared error: {rmse}, R-Squared: {r_squared}")


lgm_r = lgm.LGBMRegressor()
pipe_lgm_r = Pipeline([('scaler', StandardScaler()), ('model', lgm_r)])
pipe_lgm_r.fit(X_train, y_train)
y_pred = pipe_lgm_r.predict(X_test)
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = metrics.mean_squared_error(y_pred, y_test, squared=False)
r_squared = metrics.r2_score(y_pred, y_test)

test_copy.loc[:, 'prediction'] = pipe_lgm_r.predict(test_copy)
test_copy = test_series.merge(test_copy[['prediction']], how='left', left_index=True, right_index=True)
plt.figure(figsize=(15, 10))
ax1 = sns.kdeplot(test_copy['prediction'], color="r", label="Actual Value")
sns.kdeplot(test_copy['timeseries-tas-monthly-mean'], color="b", label="Fitted Values", ax=ax1)
plt.show()

print(f"Model: {lgm_r}, Mean squared error: {mse}, Root mean squared error: {rmse}, R-Squared: {r_squared}")
