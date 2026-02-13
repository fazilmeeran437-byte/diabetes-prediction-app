# train_multivariate_lstm.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

print("✅ Starting Multivariate LSTM training...")

# -------------------------
# 1️⃣ Load multivariate CSV
# -------------------------
df = pd.read_csv("multivariate_glucose_timeseries.csv")  # must be in same folder
print("✅ CSV loaded! First 5 rows:")
print(df.head())

# Drop Day column
data = df.drop(columns=["Day"]).values  # shape: (num_rows, 6)

# -------------------------
# 2️⃣ Scale data
# -------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler, "multivariate_lstm_scaler.pkl")
print("✅ Scaler saved as multivariate_lstm_scaler.pkl")

# -------------------------
# 3️⃣ Prepare sequences
# -------------------------
timesteps = 5  # last 5 days to predict next glucose
X, y = [], []

for i in range(len(scaled_data)-timesteps):
    X.append(scaled_data[i:i+timesteps])
    y.append(scaled_data[i+timesteps, 0])  # glucose is first column

X, y = np.array(X), np.array(y)
print(f"✅ X shape: {X.shape}, y shape: {y.shape}")

# -------------------------
# 4️⃣ Build LSTM model
# -------------------------
model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))  # output glucose
model.compile(optimizer="adam", loss="mse")
print("✅ LSTM model built!")

# -------------------------
# 5️⃣ Train model
# -------------------------
model.fit(X, y, epochs=50, batch_size=1, verbose=2)
print("✅ Training finished!")

# -------------------------
# 6️⃣ Save model
# -------------------------
model.save("multivariate_lstm.h5")
print("✅ Multivariate LSTM model saved as multivariate_lstm.h5")
print("🎉 Ready! Now your app.py can use this model for 30-day predictions.")