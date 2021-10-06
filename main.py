import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
plt.style.use('fivethirtyeight')

# GET THE STOCK QUOTE
company = input("Enter ticker symbol(from yahoo finance): ")
start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()
data_from = web.DataReader(company, 'yahoo',  start, end)
# SHOW THE DATA
print(data_from)

# VISUALIZE CLOSING ACTUAL PRICE
# plt.figure(figsize=(16,8))
# plt.title(f"{company} Close Price History")
# plt.plot(data_from['Close'])        # feeding close price data
# plt.xlabel('Time')
# plt.ylabel('Close Price')
# plt.show()

# CREATING NEW DATA FRAME WITH ONLY CLOSE PRICE
data = data_from.filter(['Close'])

# CONVERT DATA FRAME INTO NUMPY ARRAY
dataset = data.values
# GET NUMBER OF ROWS TO TRAIN THE MODEL ON
training_data_len = math.ceil(len(dataset)*0.8)
print(f"Number of data trained = ", training_data_len)

# SCALE THE DATA
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# CREATE SCALED TRAINING DATA SET
train_data = scaled_data[0:training_data_len, :]
# SPLIT DATA INTO X AND Y TRAIN DATA SETS
x_train = []      # independent training variables
y_train = []      # dependent training variables
# CREATE A LOOP
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])     # WILL HAVE 60 VALUES (0-59)
  y_train.append(train_data[i, 0])          # WILL HAVE 61ST VALUE (60)

# CONVERT X AND Y TRAIN TO NUMPY ARRAY
x_train = np.array(x_train)
y_train = np.array(y_train)

# RESHAPE THE DATA
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# BUILD LSTM MODEL (RECURRENT NEURAL NETWORK)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# COMPILE AND TRAIN THE MODEL
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=25)

# CREATE THE TESTING DATA SET
test_data = scaled_data[training_data_len - 60:]            # ARRAY OF SCALED VALUES
# CREATE DATA SET X AND Y TRAIN
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

# CONVERTING DATA TO NUMPY ARRAY
x_test = np.array(x_test)

# RESHAPING THE DATA
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# GET MODELS PREDICTED VALUE FOR TEST DATA SET
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# EVALUATE MODEL (GET ROOT MEAN SQUARE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
# print('Root mean sqaure = ', rmse)

# PLOT THE DATA
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# VISUALIZE THE DATA
plt.figure(figsize=(16, 8))
plt.title(f"{company} Close Price")
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train Data', 'Validated Data', 'Predictions'], loc='upper left')
plt.show()
