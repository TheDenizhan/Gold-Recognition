import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import tensorflow as tf
from keras.models import Model
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
test_size = df[df.Date.dt.year == 2021].shape[0]

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
history = model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.1, verbose=1)

# Regresyon algoritmalarını tanımla ve eğit
algorithms = {
    'Linear Regression': LinearRegression(),
    'Support Vector Regression': SVR(),
    'Decision Tree Regression': DecisionTreeRegressor(),
    'Random Forest Regression': RandomForestRegressor(),
    'Gradient Boosting Regression': GradientBoostingRegressor(),
    'logistic Regression': LogisticRegression()
}

for algorithm_name, algorithm in algorithms.items():
    # Modeli eğit ve tahmin yap
    algorithm.fit(X_train.reshape((X_train.shape[0], X_train.shape[1])), y_train.flatten())
    y_pred = algorithm.predict(X_test.reshape((X_test.shape[0], X_test.shape[1])))

    # Performans metriklerini hesapla
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    accuracy = 1 - mape

    # Sonuçları yazdır
    print(f'{algorithm_name} Results:')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}')
    print(f'Accuracy: {accuracy}')
    print('----------------------------------')

# Test sonuçlarını görselleştir
y_test_true = scaler.inverse_transform(y_test)
y_test_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'].iloc[:-test_size], scaler.inverse_transform(train_data), color='black', lw=2, label='Training Data')
ax.plot(df['Date'].iloc[-test_size:], y_test_true, color='blue', lw=2, label='Actual Test Data')
ax.plot(df['Date'].iloc[-test_size:], y_test_pred, color='red', lw=2, label='Predicted Test Data')
ax.set(xlabel="Date", ylabel="Price", title="Model Performance on Gold Price Prediction",facecolor='white')
ax.legend()
plt.grid(color='black', linestyle='--', linewidth=0.5)
plt.show()

# Eğitim sürecini görselleştir
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history.history['loss'], label='Train Loss', color='blue')
ax.plot(history.history['val_loss'], label='Validation Loss', color='orange')
ax.set(xlabel="Epoch", ylabel="Loss", title="Model Training and Validation Loss", facecolor='white')
ax.legend()
plt.grid(color='black', linestyle='--', linewidth=0.5)
plt.show()

# Get the last date in the existing dataset
last_date = df['Date'].max()
# Make Predictions for Extended Future Dates
future_data_extended = df.loc[:, 'Price'].values.reshape(-1, 1)
future_data_extended = scaler.transform(future_data_extended)

X_future_extended = []
for i in range(window_size, len(future_data_extended)):
    X_future_extended.append(future_data_extended[i - window_size:i, 0])

X_future_extended = np.array(X_future_extended)
X_future_extended = np.reshape(X_future_extended, (X_future_extended.shape[0], X_future_extended.shape[1], 1))

future_predictions_extended = model.predict(X_future_extended)
future_predictions_extended = scaler.inverse_transform(future_predictions_extended.reshape(-1, 1))

# Extend the time range for future predictions
future_dates_extended = pd.date_range(start=last_date, periods=len(future_predictions_extended), freq='D')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Price'], color='black', lw=2, label='Historical Data')
extended_prices = np.concatenate([np.array([df['Price'].values[-1]]), future_predictions_extended.flatten()])
min_length = min(len(future_dates_extended), len(extended_prices))
ax.plot(future_dates_extended[:min_length], extended_prices[:min_length], color='green', lw=2, label='Predicted Future Data (Extended)')
ax.set(xlabel="Date", ylabel="Price", title="Model Predictions for Gold Prices", facecolor='white')
ax.legend()
plt.grid(color='black', linestyle='--', linewidth=0.5)
plt.show()
# LSTM modelinden gelecek tahminleri al
future_predictions_lstm = model.predict(X_future_extended)
future_predictions_lstm = scaler.inverse_transform(future_predictions_lstm.reshape(-1, 1))

# Linear regresyon modelini tanımla
linear_model = LinearRegression()

# Eğitim verileri üzerinde modeli eğit
linear_model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1])), y_train.flatten())

# LSTM'den alınan gelecek tahminler için linear regresyon modeli üzerinde tahmin yap
future_predictions_linear = linear_model.predict(X_future_extended.reshape((X_future_extended.shape[0], X_future_extended.shape[1])))

# Tahminleri orijinal ölçeklendirmeye döndür
future_predictions_linear = scaler.inverse_transform(future_predictions_linear.reshape(-1, 1))

# Görselleştirme
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Price'], color='black', lw=2, label='Historical Data')
ax.plot(future_dates_extended, future_predictions_lstm, color='red', lw=2, label='LSTM Predicted Future Data')
ax.plot(future_dates_extended, future_predictions_linear, color='blue', lw=2, label='Linear Regression Predicted Future Data')
ax.set(xlabel="Date", ylabel="Price", title="Model Predictions for Gold Prices", facecolor='white')
ax.legend()
plt.grid(color='black', linestyle='--', linewidth=0.5)
plt.show()
