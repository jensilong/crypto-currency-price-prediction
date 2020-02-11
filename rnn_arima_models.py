import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data(filename):
    return pd.read_csv(filename)

def rnn_preprocessing_data(dataset):
    # use timestamp to add additional feature date to datasets
    dataset['date'] = pd.to_datetime(dataset['Timestamp'], unit='s').dt.date

    # Filter/group datasets by dates
    dt = dataset.groupby('date')

    # Calculate the mean for weighted price for each crypto exchange
    btc_usd_price_mean = dt['Weighted_Price'].mean()

    return btc_usd_price_mean

def split_data(btc_usd_price_mean):

    prediction_period = 30
    training_df = btc_usd_price_mean[len(btc_usd_price_mean)- prediction_period:]
    test_df = btc_usd_price_mean[:len(btc_usd_price_mean)- prediction_period]

    training_dataset = training_df.values
    training_dataset = np.reshape(training_dataset, (len(training_dataset), 1))

    scaler = MinMaxScaler()

    training_dataset = scaler.fit_transform(training_dataset)
    x_train = training_dataset[0:len(training_dataset)-1]
    x_train = np.reshape(x_train, (len(x_train), 1, 1))
    y_train = training_dataset[1:len(training_dataset)]

    return x_train, y_train, test_df

def run_rnn_model(x_train, y_train, test_df):

    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    rnn = Sequential()
    rnn.add(LSTM(units = 500, activation= 'sigmoid', input_shape = (None, 1)))
    rnn.add(Dense(units = 1))
    rnn.compile(optimizer = 'adam', loss = 'mae')
    rnn.fit(x_train, y_train, batch_size=5, epochs=100)

    test_dataset = test_df.values[1:]
    scaler = MinMaxScaler()
    btc_prices = np.reshape(test_df.values[0:len(test_df)-1], (len(test_dataset), 1))
    btc_prices = scaler.fit_transform(btc_prices)
    btc_prices = np.reshape(btc_prices, (len(btc_prices), 1, 1))
    predicted = rnn.predict(btc_prices)
    predicted = scaler.inverse_transform(predicted)

    return predicted, test_dataset

def display_results(title, test_dataset, test_df, predicted):
    plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    plt.plot(test_dataset, color = 'red', label = 'Actual BTC Price')
    plt.plot(predicted, color = 'green', label = 'Predicted BTC Price')
    plt.title(title, fontsize=30)
    test_df = test_df.reset_index()
    x =test_df.index
    labels = test_df['date']
    plt.xticks(x, labels, rotation = 'vertical')

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(15)

    plt.xlabel('Time', fontsize=30)
    plt.ylabel('BTC Price(USD)', fontsize=30)
    plt.legend(loc=3, prop={'size': 20})
    plt.show()

def  main():

    coinbase_file = "../data/Coinbase_BTCUSD_1M_2014-12-01_to_2019-01-09.csv"
    dataset = load_data(coinbase_file)
    btc_usd_price_mean = rnn_preprocessing_data(dataset)
    x_train, y_train, test_df = split_data(btc_usd_price_mean)
    rnn_predicted_coinbase_btc_prices, test_dataset = run_rnn_model(x_train,y_train,test_df)
    display_results('Coinbase BTC/USD Price Prediction',test_dataset,test_df, rnn_predicted_coinbase_btc_prices)


if __name__ == "__main__":
    main()

