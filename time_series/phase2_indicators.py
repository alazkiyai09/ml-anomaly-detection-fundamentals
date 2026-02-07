import csv
import os
import requests
import pandas as pd
import psycopg2
from io import StringIO
from psycopg2 import sql
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from telegram import Bot
import asyncio
import numpy as np
import talib
import mplfinance as mpf
import matplotlib.dates as mdates
import arch
from matplotlib.dates import DateFormatter
import yfinance as yf
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import clone_model
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, GRU, Transformer, MaxPooling1D, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import RootMeanSquaredError
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMAResults
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import pickle

def create_connection():
    # SECURITY WARNING: Use environment variables for credentials in production
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'Stock_Data'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD')  # Required: must be set in environment
    )
    return conn

def read_table(table_name):
    # Establish a connection to the PostgreSQL database
    conn = create_connection()
    cursor = conn.cursor()

    # Define the table name in the PostgreSQL database

    # Execute a SELECT query to fetch the data from the table
    # Use sql.Identifier to prevent SQL injection
    cursor.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name)))

    # Fetch all the rows from the result set
    rows = cursor.fetchall()

    # Get the column names from the cursor description
    columns = [desc[0] for desc in cursor.description]

    # Create a DataFrame from the fetched data and column names
    data = pd.DataFrame(rows, columns=columns)

    # Print the DataFrame or perform further operations

    # Close the cursor and the database connection
    cursor.close()
    conn.close()

    return data

def fill_date(stock_data):
    start_date = stock_data['timestamp'].iloc[0]
    end_date = stock_data['timestamp'].iloc[-1]
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    stock_data = stock_data.set_index('timestamp')
    stock_data = stock_data.reindex(date_range)
    stock_data = stock_data.fillna(method='ffill')
    return stock_data

def check_index(df1, df2, df3):
    # Compare the index values of df1 and df2
    if df1.index.equals(df2.index):
        print("df1 and df2 have the same index values.")
    else:
        print("df1 and df2 have different index values.")

    # Compare the index values of df1 and df3
    if df1.index.equals(df3.index):
        print("df1 and df3 have the same index values.")
    else:
        print("df1 and df3 have different index values.")

    # Compare the index values of df2 and df3
    if df2.index.equals(df3.index):
        print("df2 and df3 have the same index values.")
    else:
        print("df2 and df3 have different index values.")


def merge_dataframe(df1, df2, df3):
    merged_df = df1.merge(df2, left_index=True, right_index=True)

    # Merge the merged_df with df3 based on their indexes
    final_merged_df = merged_df.merge(df3, left_index=True, right_index=True)
    return final_merged_df

def preprocessing_data(stock_name, maturity_1, maturity_2, volatility, usdx):
    table_name = stock_name.lower() + "_daily_data"
    stock_data = read_table(table_name)


    table_name = "treasury" + maturity_1.lower() +"_daily_data"
    bond_data = read_table(table_name)

    table_name = "treasury" + maturity_2.lower() +"_daily_data"
    bond_data_2 = read_table(table_name)

    table_name = volatility.lower() + "_daily_data"
    vix_data = read_table(table_name)

    start_date = stock_data['timestamp'].iloc[0]
    end_date = stock_data['timestamp'].iloc[-2]

    table_name = usdx.lower() + "_daily_data"
    table_name = table_name.replace('.', '')
    table_name = table_name.replace('-', '')
    dxy_data = read_table(table_name)

    stock_data['adjusted_close'] = pd.to_numeric(stock_data['adjusted_close'], errors='coerce')
    stock_data['high'] = pd.to_numeric(stock_data['high'], errors='coerce')
    stock_data['low'] = pd.to_numeric(stock_data['low'], errors='coerce')
    bond_data['value'] = pd.to_numeric(bond_data['value'], errors='coerce')
    
    #Calculate Returns
    stock_returns = np.log(stock_data['adjusted_close'] / stock_data['adjusted_close'].shift(1)).fillna(0)
    bond_returns = np.log(bond_data['value']).diff().fillna(0)

    #Multivariate Garch Model
    returns = pd.concat([stock_returns, bond_returns], axis=1).dropna().iloc[:, 0]
    model = arch.arch_model(returns, vol='Garch', p=1, q=1, rescale=False, dist='normal')
    model_fit = model.fit()
    forecast_volatility = model_fit.forecast(start=0, reindex=True)
    stock_data['MGARCH'] =  np.sqrt(forecast_volatility.variance.values)

    # Calculate RSI
    stock_data['RSI'] = talib.RSI(stock_data['adjusted_close'], timeperiod=14)
    stock_data['RSI'] = stock_data['RSI'].fillna(0)

    # Calculate ADX 
    stock_data['ADX'] = talib.ADX(stock_data['high'], stock_data['low'], stock_data['adjusted_close'], timeperiod=20)
    stock_data['ADX'] = stock_data['ADX'].fillna(0)
 
    stock_data = fill_date(stock_data)
    bond_data = fill_date(bond_data)
    bond_data_2 = fill_date(bond_data_2)
    vix_data = fill_date(vix_data)
    dxy_data = fill_date(dxy_data)

    bond_data = bond_data.merge(bond_data_2, left_index=True, right_index=True)
    price_change = stock_data['adjusted_close'].diff() / stock_data['adjusted_close'].shift(1) * 100

    stock_data['price_change'] = price_change
    stock_data['price_change'] = stock_data['price_change'].fillna(0)

    bond_data = bond_data[(bond_data.index >= start_date) & (bond_data.index <= end_date)]
    stock_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)]
    vix_data = vix_data[(vix_data.index >= start_date) & (vix_data.index <= end_date)]
    dxy_data = dxy_data[(dxy_data.index >= start_date) & (dxy_data.index <= end_date)]

    bond_data['DXY'] = dxy_data['adjusted_close']

    check_index(stock_data, bond_data, vix_data)
    merged_data = merge_dataframe(stock_data, bond_data, vix_data)
    merged_data = merged_data[['adjusted_close_x', 'volume_x','adjusted_close_y', 'value_x', 'value_y', 'RSI', 'MGARCH', 'ADX', 'DXY']]

    return merged_data

def LSTM_process(data, stock_name):
    target = data['adjusted_close_x'].values
    features = data.drop(['adjusted_close_x'], axis=1)
    volume = data['volume_x'].values

    log_transformed_volume = np.log1p(np.array(volume, dtype=np.float64))
    numeric_features = np.array(features.drop(['volume_x'], axis=1).values, dtype=np.float64)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    log_transformed_volume = np.reshape(log_transformed_volume, (-1, 1))  # Reshape to a 2D array with a single column
    scaled_features = np.concatenate((scaled_features, log_transformed_volume), axis=1)  # Concatenate along the columns

    # Define the number of previous days to consider
    n_prev_days = 60

    # Prepare the input sequences
    input_sequences = []
    for i in range(n_prev_days, len(data)):
        input_sequences.append(scaled_features[i-n_prev_days:i, :])

    # Convert to numpy array
    input_sequences = np.array(input_sequences)

    # Define the number of folds for cross-validation
    n_splits = 5

    # Initialize the cross-validation
    kfold = TimeSeriesSplit(n_splits=n_splits)
    k = 0
    # Iterate over the folds
    for train_index, test_index in kfold.split(input_sequences):
        # Split the data into training and test sets for the current fold
        X_train, X_test = input_sequences[train_index], input_sequences[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # Convert the input arrays to TensorFlow tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

        # Define the learning rate
        learning_rate = 0.002
        reg = regularizers.l1_l2(l1=0.01, l2=0.01)
        # Initialising the RNN
        model = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        model.add(LSTM(units=128, return_sequences = True, kernel_regularizer=reg, input_shape=(n_prev_days, features.shape[1])))
        model.add(Dropout(0.1))

        # Adding a second LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64,activation='relu', return_sequences = True, kernel_regularizer=reg))
        model.add(Dropout(0.2))
        
        # Adding a second LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64,activation='relu', return_sequences = True, kernel_regularizer=reg))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 32, activation='tanh', return_sequences = True, kernel_regularizer=reg))
        model.add(Dropout(0.4))

        # Adding a third LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 16, activation='linear', kernel_regularizer=reg))

        # Adding the output layer
        model.add(Dense(units = 1,activation='linear', kernel_regularizer=reg))
        # Print the model summary
        model.summary()
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[RootMeanSquaredError()])

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='root_mean_squared_error', patience=10, min_delta=0.0001, restore_best_weights=True, mode="min")
        model_checkpoint = ModelCheckpoint(stock_name+'_LSTM_'+str(k)+'.h5', monitor='root_mean_squared_error', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

        # Train the model with early stopping
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint, reduce_lr])
        k+=1
        print(k)

    # Make predictions for the next 7 consecutive days using the best model

    #last_60_days = np.reshape(last_60_days, (last_60_days.shape[0], last_60_days.shape[1]))


def GRU_process(data, stock_name):
    target = data['adjusted_close_x'].values
    features = data.drop(['adjusted_close_x'], axis=1)
    volume = data['volume_x'].values

    log_transformed_volume = np.log1p(np.array(volume, dtype=np.float64))
    numeric_features = np.array(features.drop(['volume_x'], axis=1).values, dtype=np.float64)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    log_transformed_volume = np.reshape(log_transformed_volume, (-1, 1))  # Reshape to a 2D array with a single column
    scaled_features = np.concatenate((scaled_features, log_transformed_volume), axis=1)  # Concatenate along the columns

    # Define the number of previous days to consider
    n_prev_days = 60

    # Prepare the input sequences
    input_sequences = []
    for i in range(n_prev_days, len(data)):
        input_sequences.append(scaled_features[i-n_prev_days:i, :])

    # Convert to numpy array
    input_sequences = np.array(input_sequences)

    # Define the number of folds for cross-validation
    n_splits = 5

    # Initialize the cross-validation
    kfold = TimeSeriesSplit(n_splits=n_splits)
    k = 0
    # Iterate over the folds
    for train_index, test_index in kfold.split(input_sequences):
        # Split the data into training and test sets for the current fold
        X_train, X_test = input_sequences[train_index], input_sequences[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # Convert the input arrays to TensorFlow tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

        # Define the learning rate
        learning_rate = 0.002
        reg = regularizers.l1_l2(l1=0.01, l2=0.01)
        gru_model = Sequential()
        gru_model.add(GRU(units=128, return_sequences=True, input_shape=(n_prev_days, features.shape[1]), kernel_regularizer=reg))
        gru_model.add(Dropout(0.2))
        gru_model.add(GRU(units=64, return_sequences=False, kernel_regularizer=reg))
        gru_model.add(Dropout(0.2))
        gru_model.add(Dense(units=1))

        gru_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=[RootMeanSquaredError()])

        # Define model checkpoint for GRU
        gru_checkpoint = ModelCheckpoint(stock_name+'GRU_model'+str(k)+'.h5', monitor='val_root_mean_squared_error', save_best_only=True)
        early_stopping = EarlyStopping(monitor='root_mean_squared_error', patience=10, min_delta=0.0001, restore_best_weights=True, mode="min")
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        # Train the GRU model
        gru_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, gru_checkpoint, reduce_lr])
        k+=1

def transformer_process(data, stock_name):
    target = data['adjusted_close_x'].values
    features = data.drop(['adjusted_close_x'], axis=1)
    volume = data['volume_x'].values

    log_transformed_volume = np.log1p(np.array(volume, dtype=np.float64))
    numeric_features = np.array(features.drop(['volume_x'], axis=1).values, dtype=np.float64)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    log_transformed_volume = np.reshape(log_transformed_volume, (-1, 1))  # Reshape to a 2D array with a single column
    scaled_features = np.concatenate((scaled_features, log_transformed_volume), axis=1)  # Concatenate along the columns

    # Define the number of previous days to consider
    n_prev_days = 60

    # Prepare the input sequences
    input_sequences = []
    for i in range(n_prev_days, len(data)):
        input_sequences.append(scaled_features[i-n_prev_days:i, :])

    # Convert to numpy array
    input_sequences = np.array(input_sequences)

    # Define the number of folds for cross-validation
    n_splits = 5

    # Initialize the cross-validation
    kfold = TimeSeriesSplit(n_splits=n_splits)
    k = 0
    # Iterate over the folds
    for train_index, test_index in kfold.split(input_sequences):
        # Split the data into training and test sets for the current fold
        X_train, X_test = input_sequences[train_index], input_sequences[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # Convert the input arrays to TensorFlow tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


        # Build the transformer model
        learning_rate = 0.002
        reg = regularizers.l1_l2(l1=0.01, l2=0.01)
        transformer_model = Sequential()
        transformer_model.add(Transformer(num_layers=2, d_model=128, num_heads=8, dff=512, input_vocab_size=8500, maximum_position_encoding=10000))
        transformer_model.add(Dense(units=1))

        transformer_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=[RootMeanSquaredError()])

        # Define model checkpoint for Transformer
        transformer_checkpoint = ModelCheckpoint(stock_name+'transformer_model'+str(k)+'.h5', monitor='val_root_mean_squared_error', save_best_only=True)
        early_stopping = EarlyStopping(monitor='root_mean_squared_error', patience=10, min_delta=0.0001, restore_best_weights=True, mode="min")
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        # Train the Transformer model
        transformer_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, transformer_checkpoint, reduce_lr])


def prediction_result(data,n, model_name):
    n_prev_days = 60
    model = load_model(model_name)

    target = data['price_change'].values
    features = data.drop(['price_change'], axis=1)
    volume = data['volume_x'].values

    log_transformed_volume = np.log1p(np.array(volume, dtype=np.float64))
    numeric_features = np.array(features.drop(['volume_x'], axis=1).values, dtype=np.float64)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    log_transformed_volume = np.reshape(log_transformed_volume, (-1, 1))  # Reshape to a 2D array with a single column
    scaled_features = np.concatenate((scaled_features, log_transformed_volume), axis=1)  # Concatenate along the columns

    next_n_days = []
    last_price = float(data['adjusted_close_x'].iloc[-1])
    print(last_price)
    last_date = data.index[-1]
    print(last_date)
    future_dates = pd.date_range(start=last_date, periods=n, freq='D')
    for i in range(0, n):
        last_60_days = scaled_features[-(n_prev_days+(n-i)):len(scaled_features)-(n-i), :]
        prediction = model.predict(np.array([last_60_days]))
        temp = prediction[0, 0]
        # print(temp)
        # intermediate_result = temp * last_price
        # predicted_price = intermediate_result / 100 + last_price
        # last_price = predicted_price
        next_n_days.append([future_dates[i], temp])
    
    print(next_n_days)


def CNNLSTM_process(data, stock_name):
    target = data['adjusted_close_x'].values
    features = data.drop(['adjusted_close_x'], axis=1)
    volume = data['volume_x'].values

    log_transformed_volume = np.log1p(np.array(volume, dtype=np.float64))
    numeric_features = np.array(features.drop(['volume_x'], axis=1).values, dtype=np.float64)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    log_transformed_volume = np.reshape(log_transformed_volume, (-1, 1))  # Reshape to a 2D array with a single column
    scaled_features = np.concatenate((scaled_features, log_transformed_volume), axis=1)  # Concatenate along the columns

    # Define the number of previous days to consider
    n_prev_days = 60

    # Prepare the input sequences
    input_sequences = []
    for i in range(n_prev_days, len(data)):
        input_sequences.append(scaled_features[i-n_prev_days:i, :])

    # Convert to numpy array
    input_sequences = np.array(input_sequences)

    # Define the number of folds for cross-validation
    n_splits = 5

    # Initialize the cross-validation
    kfold = TimeSeriesSplit(n_splits=n_splits)
    k = 0
    # Iterate over the folds
    for train_index, test_index in kfold.split(input_sequences):
        # Split the data into training and test sets for the current fold
        X_train, X_test = input_sequences[train_index], input_sequences[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # Convert the input arrays to TensorFlow tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

        # Define the learning rate
        learning_rate = 0.002
        reg = regularizers.l1_l2(l1=0.01, l2=0.01)

        # Initialising the RNN
        model = Sequential()

        # Adding the 1D Convolutional layer
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_prev_days, features.shape[1])))
        model.add(MaxPooling1D(pool_size=2))

        # Adding the first LSTM layer and some Dropout regularisation
        model.add(LSTM(units=128, return_sequences=True, kernel_regularizer=reg))
        model.add(Dropout(0.1))

        # Adding a second LSTM layer and some Dropout regularisation
        model.add(LSTM(units=64, activation='relu', return_sequences=True, kernel_regularizer=reg))
        model.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        model.add(LSTM(units=32, activation='tanh', return_sequences=True, kernel_regularizer=reg))
        model.add(Dropout(0.4))

        # Adding a fourth LSTM layer and some Dropout regularisation
        model.add(LSTM(units=16, activation='linear', kernel_regularizer=reg))

        # Adding the output layer
        model.add(Dense(units=1, activation='linear', kernel_regularizer=reg))

        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[RootMeanSquaredError()])

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_root_mean_squared_error', patience=10, min_delta=0.0001, restore_best_weights=True, mode="min")
        model_checkpoint = ModelCheckpoint(stock_name+'_CNNLSTM_'+str(k)+'.h5', monitor='val_root_mean_squared_error', save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

        # Train the model with early stopping
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint, reduce_lr])
        k+=1

    # Make predictions for the next 7 consecutive days using the best model

    #last_60_days = np.reshape(last_60_days, (last_60_days.shape[0], last_60_days.shape[1]))

def ARIMA_process(data, stock_name):
    # Use auto_arima for parameter selection
    model = auto_arima(
        data['adjusted_close_x'], exogenous=data[['adjusted_close_y', 'value_x', 'value_y', 'RSI', 'MGARCH', 'ADX', 'DXY']],
        start_p=0, start_q=0,
        test='kpss',       # use adftest to find optimal 'd'
        max_p=5, max_q=5, # maximum p and q
        m=1,              # frequency of series
        d=None,           # let model determine 'd'
        seasonal=False,   # No Seasonality
        start_P=0, 
        D=0, 
        trace=True,
        error_action='ignore',  
        suppress_warnings=True, 
        stepwise=True
    )
    print(model.summary())
    # Fit the ARIMA model with the best parameters
    model_fit = model.fit(data['adjusted_close_x'], exogenous=data[['adjusted_close_y', 'value_x', 'value_y', 'RSI', 'MGARCH', 'ADX', 'DXY']])
    # Save the ARIMA model
    with open(stock_name+'ARIMA.pkl', 'wb') as f:
        pickle.dump(model_fit, f)


def prediction_ARIMA(data, n, model_name):
    with open(model_name, 'rb') as f:
        model_fit = pickle.load(f)

    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=n, freq='D')
    future_exog = data.iloc[-n:][['adjusted_close_y', 'value_x', 'value_y', 'RSI', 'MGARCH', 'ADX', 'DXY']]

    future_predictions = model_fit.predict(n_periods=n, exogenous=future_exog)

    next_n_days = []
    for date, prediction in zip(future_dates, future_predictions):
        #print(f"Date: {date.date()}, Predicted Price: {prediction}")
        next_n_days.append([date, prediction])
    print(next_n_days)

def ensemble_prediction(data, n, model_names):

    # Make predictions using each model
    predictions_ARIMA = prediction_ARIMA(data, n, model_names)
    predictions_LSTM = prediction_result(data, n, model_names)
    predictions_GRU = prediction_result(data, n, model_names)
    predictions_Transformer = prediction_result(data, n, model_names)
    predictions = [predictions_ARIMA, predictions_LSTM, predictions_GRU, predictions_Transformer]

    # Average the predictions
    avg_prediction = np.mean(predictions, axis=0)

    # Return the averaged predictions
    return avg_prediction


async def main():
    stock_name = 'NVDA'
    maturity_1 = '2year'
    maturity_2 = '10year'
    volatility = 'VIX'
    usdx = 'DX-Y.NYB'
    merged_data = preprocessing_data(stock_name, maturity_1, maturity_2, volatility, usdx)
    print(merged_data)
    LSTM_process(merged_data, stock_name)
    #ARIMA_process(merged_data, stock_name)
    #prediction_LSTM(merged_data,30, 'NVDA_LSTM.h5')
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())