import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, LSTM, Dropout

# TensorFlow'un GPU kullanıp kullanmadığını kontrol et
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Veri setini oku
df = pd.read_csv('Gold_Price(2013-2023).csv')

# Veri seti bilgilerini görüntüle
df.info()

# Gereksiz sütunları kaldır
df.drop(['Vol.', 'Change %'], axis=1, inplace=True)

# Tarih sütununu datetime formatına çevir ve sırala
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# Sayısal sütunlardaki virgülleri kaldır
NumCols = df.columns.drop(['Date'])
df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
df[NumCols] = df[NumCols].astype('float64')

# Tarih sütunundaki tekrar eden satırları kontrol et
df.duplicated().sum()

# Eksik değerleri kontrol et
df.isnull().sum().sum()

# Veriyi görselleştir
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.Date, df.Price, color='black', lw=2)
ax.set(xlabel="Tarih", ylabel="Fiyatı", title="Altın Fiyatı Tarihi", facecolor='white')
plt.grid(color='black', linestyle='--', linewidth=0.5)
plt.show()

# Test verisi boyutunu belirle
test_size = df[df.Date.dt.year == 2019].shape[0]

# Eğitim ve test setlerini görselleştir
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.Date[:-test_size], df.Price[:-test_size], color='green', lw=2, label='Eğitilmiş Veri')
ax.plot(df.Date[-test_size:], df.Price[-test_size:], color='red', lw=2, label='Test Verisi')
ax.set(xlabel="Tarih", ylabel="Fiyat", title="Altın Fiyatı Eğitim ve Test", facecolor='white')
ax.legend()
plt.grid(color='black', linestyle='--', linewidth=0.5)
plt.show()

# Veriyi normalize et
scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1, 1))

# Hareketli pencere boyutunu belirle
window_size = 60

# Eğitim veri setini oluştur
train_data = df.Price[:-test_size]
train_data = scaler.transform(train_data.values.reshape(-1, 1))

X_train, y_train = [], []

for i in range(window_size, len(train_data)):
    X_train.append(train_data[i - window_size:i, 0])
    y_train.append(train_data[i, 0])

# Test veri setini oluştur
test_data = df.Price[-test_size - window_size:]
test_data = scaler.transform(test_data.values.reshape(-1, 1))

X_test, y_test = [], []

for i in range(window_size, len(test_data)):
    X_test.append(test_data[i - window_size:i, 0])
    y_test.append(test_data[i, 0])

# Veriyi uygun hale getir
X_train, X_test, y_train, y_test = map(np.array, [X_train, X_test, y_train, y_test])

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

# Modeli tanımla
def define_model():
    input1 = Input(shape=(window_size, 1))
    x = LSTM(units=128, return_sequences=True)(input1)
    x = Dropout(0.2)(x)
    x = LSTM(units=64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=32)(x)
    x = Dropout(0.2)(x)
    dnn_output = Dense(1)(x)

    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam')
    model.summary()

    return model

# Modeli eğit
model = define_model()
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.1, verbose=1)

# Model performansını değerlendir
result = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

MAPE = mean_absolute_percentage_error(y_test, y_pred)
Accuracy = 1 - MAPE

# Test sonuçlarını görselleştir
y_test_true = scaler.inverse_transform(y_test)
y_test_pred = scaler.inverse_transform(y_pred)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'].iloc[:-test_size], scaler.inverse_transform(train_data), color='black', lw=2, label='Training Data')
ax.plot(df['Date'].iloc[-test_size:], y_test_true, color='blue', lw=2, label='Actual Test Data')
ax.plot(df['Date'].iloc[-test_size:], y_test_pred, color='red', lw=2, label='Predicted Test Data')
ax.set(xlabel="Date", ylabel="Price", title="Model Performance on Gold Price Prediction", facecolor='yellow')
ax.legend()
plt.grid(color='white', linestyle='--', linewidth=0.5)
plt.show()

# Eğitim sürecini görselleştir
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history.history['loss'], label='Train Loss', color='blue')
ax.plot(history.history['val_loss'], label='Validation Loss', color='orange')
ax.set(xlabel="Epoch", ylabel="Loss", title="Model Training and Validation Loss", facecolor='white')
ax.legend()
plt.grid(color='black', linestyle='--', linewidth=0.5)
plt.show()
