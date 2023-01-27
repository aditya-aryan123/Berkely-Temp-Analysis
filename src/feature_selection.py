import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

df_cru = pd.read_csv('../input/CRU.csv')
df_cru.drop('time', axis=1, inplace=True)

train_series = df_cru.loc[(df_cru['year'] >= 1900) & (df_cru['year'] < 2010), :]
test_series = df_cru.loc[(df_cru['year'] >= 2010), :]

test_copy = test_series.drop('timeseries-tas-monthly-mean', axis=1)

X = train_series.drop('timeseries-tas-monthly-mean', axis=1)
y = train_series['timeseries-tas-monthly-mean'].values

reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" % reg.score(X, y))
coef = pd.DataFrame(reg.coef_, X.columns)
coef = coef.reset_index()
coef.columns = ['Features', 'Coefficient']
coef.to_csv('coef.csv', index=False)
