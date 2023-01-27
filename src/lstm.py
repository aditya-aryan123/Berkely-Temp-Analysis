import pandas as pd
import torch
import os.path as osp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler

df_cru = pd.read_csv('../input/CRU_Updated.csv')
df_cru.drop('time', axis=1, inplace=True)

train_series = df_cru.loc[(df_cru['year'] >= 1900) & (df_cru['year'] < 2010), :]
test_series = df_cru.loc[(df_cru['year'] >= 2010), :]

test_copy = test_series.drop('timeseries-tas-monthly-mean', axis=1)

scaler = StandardScaler()
scaler.fit(train_series)
scaled_train = scaler.transform(train_series)


def df_to_x_y(df, window_size=1):
    X = []
    y = []
    for i in range(len(df) - window_size):
        row = [[a] for a in df[i: i + window_size, 0]]
        X.append(row)
        label = df[i + window_size, 0]
        y.append(label)
    return np.array(X), np.array(y)


time_steps = 60
X_1, y_1 = df_to_x_y(scaled_train, time_steps)

X_train, y_train = X_1[:998], y_1[:998]
X_test, y_test = X_1[998:], y_1[998:]

model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, shuffle=False)

early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True,
)

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(loc="best")
plt.xlabel("No. Of Epochs")
plt.ylabel("mse score")
plt.show()
