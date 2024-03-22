# Necessary Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
import math
from sklearn.metrics import mean_squared_error

# Part 1 - Data preprocessing
def preprocess_data(file_path):
    training_set = pd.read_csv(file_path)
    values = training_set.iloc[:, 1:2].values
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)
    return scaler, values_scaled

file_path_train = "Stock_Train_Data.csv"
scaler, training_set_scaled = preprocess_data(file_path_train)

# Input and Output for the network
X_train = training_set_scaled[0:1257]
y_train = training_set_scaled[1:1258]

# Reshape the dataset
X_train = np.reshape(X_train, (1257, 1, 1))

# Part 2 - Hyperparameter Tuning
class RegressionHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units', min_value=4, max_value=64, step=4),
                       activation='sigmoid', input_shape=self.input_shape))
        model.add(Dense(units=1))
        model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop']),
                      loss='mean_squared_error')
        return model

input_shape = (1, 1)
hypermodel = RegressionHyperModel(input_shape=input_shape)

tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    directory='random_search',
    project_name='stock_price'
)

tuner.search(X_train, y_train, epochs=5, validation_split=0.2, verbose=1)

# Get the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]

# Train the best model further (if needed)
best_model.fit(X_train, y_train, batch_size=32, epochs=200)

# Part 3 - Building the static RNN model
def build_static_rnn_model(input_shape):
    regressor = Sequential()
    regressor.add(LSTM(units=4, activation='sigmoid', input_shape=input_shape))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor

regressor_static = build_static_rnn_model((None, 1))
regressor_static.fit(X_train, y_train, batch_size=32, epochs=200)

# Part 4 - Making predictions using best_model
def make_predictions(model, inputs, scaler):
    inputs_scaled = scaler.transform(inputs)
    inputs_reshaped = np.reshape(inputs_scaled, (len(inputs), 1, 1))
    predicted_stock_price = model.predict(inputs_reshaped)
    return scaler.inverse_transform(predicted_stock_price)

file_path_test = "Stock_Test_Data.csv"
test_set = pd.read_csv(file_path_test)
real_stock_price = test_set.iloc[:, 1:2].values

predicted_stock_price_best_model = make_predictions(best_model, real_stock_price, scaler)
predicted_stock_price_regressor_static = make_predictions(regressor_static, real_stock_price, scaler)

# Part 5 - Visualising the results for both models  
plt.figure(figsize=(15, 6))

plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price_best_model, color='blue', label='Predicted (Tuned) Google Stock Price')
plt.plot(predicted_stock_price_regressor_static, color='green', label='Predicted (Static) Google Stock Price')

plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# RMSE for both models
rmse_best_model = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price_best_model))
rmse_regressor_static = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price_regressor_static))

print(f"RMSE for Best Model: {rmse_best_model}")
print(f"RMSE for Static Regressor: {rmse_regressor_static}")
