import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from keras.models import load_model
import streamlit as st

ticker = st.text_input('Enter stock ticker: ', 'TSLA')
df = yf.download(ticker, start="2012-01-01")
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'], dayfirst = True)

st.subheader('Previous Data from 2012 to present: ')
st.text(df.describe())

st.subheader('Time vs Closing:')
fig = plt.figure(figsize=(10, 5))
plt.plot(df.Close)
st.pyplot(fig)

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

st.subheader('Time vs Closing along with moving avg of 100 and 200 days:')
fig = plt.figure(figsize=(10, 5))
plt.plot(df.Close)
plt.plot(ma100, 'r') 
plt.plot(ma200, 'g')
st.pyplot(fig)

train = pd.DataFrame(df['Close'][:int(len(df)*0.70)])
test = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range = (0, 1))

train_arr = scaler.fit_transform(train)

X_train = []
Y_train = []
for i in range(100, train_arr.shape[0]):
    X_train.append(train_arr[i-100:i])
    Y_train.append(train_arr[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

model = load_model('keras_model.h5')

prev100 = train.tail(100)
final_df = pd.concat([prev100, test], ignore_index = True)
input_data = scaler.fit_transform(final_df)
X_test = []
Y_test = []
for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    Y_test.append(input_data[i, 0])

X_test, Y_test = np.array(X_test), np.array(Y_test)
Y_predict = model.predict(X_test)

scale_factor = scaler.scale_[0]
Y_predict = Y_predict/scale_factor
Y_test = Y_test/scale_factor

st.subheader('Time vs Closing and predicted:')
fig = plt.figure(figsize=(12, 6))

plt.plot(Y_test, 'b', label = 'Original Price')
plt.plot(Y_predict, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig)