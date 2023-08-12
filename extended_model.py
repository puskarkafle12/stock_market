
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, LSTM
from keras.optimizers import RMSprop

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(units=75, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=20))
    model.add(Dense(units=1))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
    return model

def build_stacked_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=75, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=20))
    model.add(Dense(units=1))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
    return model

from keras.layers import SimpleRNN

def build_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2, name = 'd1'))
    model.add(SimpleRNN(units=75, return_sequences=False))
    model.add(Dropout(0.2, name = 'd2'))
    model.add(Dense(units=20))
    model.add(Dense(units=1))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
    return model
