import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

data = pd.read_csv("glucose_timeseries.csv")
glucose = data["Glucose"].values.reshape(-1, 1)

scaler = MinMaxScaler()
glucose_scaled = scaler.fit_transform(glucose)

X = []
y = []

for i in range(10, len(glucose_scaled)):
    X.append(glucose_scaled[i-10:i])
    y.append(glucose_scaled[i])

X = np.array(X)
y = np.array(y)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1],1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=8)

model.save("lstm_model.h5")
pickle.dump(scaler, open("lstm_scaler.pkl", "wb"))

print("LSTM Model Saved!")