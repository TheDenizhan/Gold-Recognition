import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras import Model
from keras import Model
from keras.layers import Input, Dense, Dropout, LSTM

# Veriyi yükle
df = pd.read_csv('Gold_Price(2013-2023).csv')
df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# Veriyi görselleştir
fig = plt.figure(figsize=(15, 6), dpi=150)
plt.plot(df.Date, df.Price, color='black', lw=2)
plt.title('Gold Price History Data', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Scaled Price', fontsize=12)
plt.show()

# Veriyi hazırla
scaler = MinMaxScaler()
df['Price'] = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

window_size = 60
test_size = df[df.Date.dt.year == 2022].shape[0]

X, y = [], []

for i in range(window_size, len(df['Price'])):
    X.append(df['Price'][i - window_size:i].values)
    y.append(df['Price'][i])

X, y = np.array(X), np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Modeli tanımla
def define_model():
    input1 = Input(shape=(window_size, 1))
    x = LSTM(units=128, return_sequences=True)(input1)
    x = Dropout(0.2)(x)
    x = LSTM(units=64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=32)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    dnn_output = Dense(1)(x)

    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['mae', 'mse'])
    model.summary()

    return model

# Modeli eğit
model = define_model()
history = model.fit(X, y, epochs=150, batch_size=32, validation_split=0.1, verbose=1)

# Modeli değerlendir
result = model.evaluate(X, y)
y_pred = model.predict(X)

R2_score = 1 - result[1] / np.var(y)

print("Test Loss:", result[0])
print("Test MAE:", result[1])
print("Test MSE:", result[2])
print("Test R2 Score:", R2_score)

# Sonuçları görselleştir
y_true = scaler.inverse_transform(y)
y_pred = scaler.inverse_transform(y_pred)

plt.figure(figsize=(15, 6), dpi=150)
plt.plot(df['Date'].iloc[:-test_size], scaler.inverse_transform(df['Price'][:-test_size].values.reshape(-1, 1)),
         color='black', lw=2)
plt.plot(df['Date'].iloc[-test_size:], y_true[-test_size:], color='blue', lw=2)
plt.plot(df['Date'].iloc[-test_size:], y_pred[-test_size:], color='red', lw=2)
plt.title('Model Performance on Gold Price Prediction', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(['Training Data', 'Actual Test Data', 'Predicted Test Data'], loc='upper left', prop={'size': 15})
plt.show()

# Eğitim sürecini görselleştir
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
