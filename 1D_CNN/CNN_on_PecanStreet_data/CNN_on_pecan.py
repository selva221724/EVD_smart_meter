# cnn model
import numpy as np
import pandas as pd
from keras.layers import LSTM
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

# ============ Read the Train CSV here ========================
data_path = r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\pecan_15mint_austin_data_feature_extracted.csv"
sourceData = pd.read_csv(data_path)

# sourceData['time_stamp'] = pd.to_datetime(sourceData['time_stamp'])

X = sourceData[['current']]  # independent Variable
Y = sourceData[['EV_label']]  # target variable

# split the data into train and test
trainX, testX = train_test_split(X, test_size=0.3, shuffle=False)
trainY, testY = train_test_split(Y, test_size=0.3, shuffle=False)

print("Shape of TrainX and TrainY ", trainX.shape, trainY.shape)
print("Shape of TestX and TestY ", testX.shape, testY.shape)

Xscaler = MinMaxScaler(feature_range=(0, 1))  # scale so that all the X data will range from 0 to 1
Xscaler.fit(trainX)
scaled_X_train = Xscaler.transform(trainX)
print('Scaled Train X Shape ', trainX.shape)

Yscaler = MinMaxScaler(feature_range=(0, 1))
Yscaler.fit(trainY)
scaled_y_train = trainY.to_numpy()
scaled_y_train = scaled_y_train.reshape(
    -1)  # remove the second dimension from y so the shape changes from (n,1) to (n,)
scaled_y_train = np.insert(scaled_y_train, 0, 0)
scaled_y_train = np.delete(scaled_y_train, -1)
print('Scaled Train Y Shape ', scaled_y_train.shape)

#  ================= Keras Model LSTM Build ===========================
n_input = 25  # how many samples/rows/timesteps to look in the past in order to forecast the next sample
n_features = trainX.shape[1]  # how many predictors/Xs/features we have to predict y
b_size = 60  # Number of time series samples in each batch
generator = TimeseriesGenerator(scaled_X_train, scaled_y_train, length=n_input, batch_size=b_size)

model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1, activation='sigmoid'))  # since it is a binary classification, we are calling sigmoid function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit_generator(generator, epochs=5)

# ===================== evaluation on test Data ============================

scaled_X_test = Xscaler.transform(testX)
test_generator = TimeseriesGenerator(scaled_X_test, np.zeros(len(testX)), length=n_input, batch_size=b_size)

y_pred_scaled = model.predict(test_generator)
y_pred = Yscaler.inverse_transform(y_pred_scaled)
results = pd.DataFrame({'y_true': testY.values[n_input:].ravel().tolist(), 'y_pred': y_pred.ravel().tolist()})
results.plot(title='Test Data - Actual vs Predicted Time Series - AC1')

plt.figure()
plt.plot(testX)
#
# # ================================== vsialization ====================================
#
scaled_X_test = Xscaler.transform(X)
test_generator = TimeseriesGenerator(scaled_X_test, np.zeros(len(X)), length=n_input, batch_size=b_size)
y_pred_scaled = model.predict(test_generator)
y_pred = Yscaler.inverse_transform(y_pred_scaled)
pred = y_pred.ravel().tolist()
pred = [-0.25 if i >= 0.5 else -0.5 for i in pred]
y_true = Y.values[n_input:].ravel().tolist()
y_true = [0.25 if i == 1 else 0 for i in y_true]
x = X['current']
normalized = (x - min(x)) / (max(x) - min(x))

results = pd.DataFrame({'y true': y_true, 'y pred': pred})
results.plot(title='Actual vs Predicted Time Series - EV Detection (PECAN STREET DATASET - 15mints Data )')
plt.plot(normalized)
plt.legend(['y true', 'y pred', 'Aggregated Current(normalized)'], loc="upper right")
