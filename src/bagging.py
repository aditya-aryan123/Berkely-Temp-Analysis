import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split


df_cru = pd.read_csv('../input/Updated_CRU.csv')
df_cru.drop('time', axis=1, inplace=True)

train_series = df_cru.loc[(df_cru['year'] >= 1900) & (df_cru['year'] < 2010), :]
test_series = df_cru.loc[(df_cru['year'] >= 2010), :]

test_copy = test_series.drop('timeseries-tas-monthly-mean', axis=1)

X = train_series.drop('timeseries-tas-monthly-mean', axis=1)
y = train_series['timeseries-tas-monthly-mean'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

dt = DecisionTreeRegressor()
pipe_dt = Pipeline([('scaler', StandardScaler()), ('model', dt)])
pipe_dt.fit(X_train, y_train)
y_pred = pipe_dt.predict(X_test)
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = metrics.mean_squared_error(y_pred, y_test, squared=False)
r_squared = metrics.r2_score(y_pred, y_test)

print(f"Model: {dt}, Mean squared error: {mse}, Root mean squared error: {rmse}, R-Squared: {r_squared}")


rf = RandomForestRegressor()
pipe_rf = Pipeline([('scaler', StandardScaler()), ('model', rf)])
pipe_rf.fit(X_train, y_train)
y_pred = pipe_rf.predict(X_test)
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = metrics.mean_squared_error(y_pred, y_test, squared=False)
r_squared = metrics.r2_score(y_pred, y_test)

print(f"Model: {rf}, Mean squared error: {mse}, Root mean squared error: {rmse}, R-Squared: {r_squared}")


etr = ExtraTreesRegressor()
pipe_etr = Pipeline([('scaler', StandardScaler()), ('model', etr)])
pipe_etr.fit(X_train, y_train)
y_pred = pipe_etr.predict(X_test)
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = metrics.mean_squared_error(y_pred, y_test, squared=False)
r_squared = metrics.r2_score(y_pred, y_test)

print(f"Model: {etr}, Mean squared error: {mse}, Root mean squared error: {rmse}, R-Squared: {r_squared}")


gbr = GradientBoostingRegressor()
pipe_gbr = Pipeline([('scaler', StandardScaler()), ('model', gbr)])
pipe_gbr.fit(X_train, y_train)
y_pred = pipe_gbr.predict(X_test)
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = metrics.mean_squared_error(y_pred, y_test, squared=False)
r_squared = metrics.r2_score(y_pred, y_test)

test_copy.loc[:, 'prediction'] = pipe_dt.predict(test_copy)
test_copy = test_series.merge(test_copy[['prediction']], how='left', left_index=True, right_index=True)
plt.figure(figsize=(15, 10))
ax1 = sns.kdeplot(test_copy['prediction'], color="r", label="Actual Value")
sns.kdeplot(test_copy['timeseries-tas-monthly-mean'], color="b", label="Fitted Values", ax=ax1)
plt.show()

print(f"Model: {gbr}, Mean squared error: {mse}, Root mean squared error: {rmse}, R-Squared: {r_squared}")


abr = AdaBoostRegressor()
pipe_abr = Pipeline([('scaler', StandardScaler()), ('model', abr)])
pipe_abr.fit(X_train, y_train)
y_pred = pipe_abr.predict(X_test)
mse = metrics.mean_squared_error(y_pred, y_test)
rmse = metrics.mean_squared_error(y_pred, y_test, squared=False)
r_squared = metrics.r2_score(y_pred, y_test)

print(f"Model: {abr}, Mean squared error: {mse}, Root mean squared error: {rmse}, R-Squared: {r_squared}")
