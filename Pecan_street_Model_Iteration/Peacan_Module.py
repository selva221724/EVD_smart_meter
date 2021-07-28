from tqdm import tqdm
from glob import glob
import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras import callbacks
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import classification_report
import os
from datetime import datetime
import logging
import tensorflow as tf


# plt.switch_backend('Agg')


class DataReader:
    """ To Read the Data and Sanitize and preprocess the Pecan Street Dataset
    """

    def __init__(self):

        pass

    @staticmethod
    def dataExtractorOneSecond(file_path, output_path):

        austineIds = [6139, 3039, 3538, 8386, 4031, 9922, 7951, 8565, 9278, 661, 7800, 9160, 8156, 7536, 2361, 2818,
                      4767, 3456, 9019, 7901, 7719, 5746, 1642, 4373, 2335]

        californiaIds = [9612, 4495, 9213, 6547, 7062, 8342, 3864, 8733, 1450, 2606, 9775, 5938, 1731, 4934, 7114, 203,
                         3938, 3687, 6377, 9836, 1524, 8061, 8574]

        newyorkIds = [387, 1417, 2318, 142, 914, 27, 3996, 3488, 558, 5679, 2096, 2358, 950, 3000, 4283, 3517, 5058,
                      4550, 1222, 5587, 1240, 9053, 5982, 5997, 3700]

        # ============================== ev present in the house ========================

        # austine = [2335, 7719, 1642, 4373, 661, 6139, 8156, 4767]
        # newyork = [5058, 1222, 5679, 3000, 27, 9053, 3517]

        dataIds = austineIds

        for id in dataIds:
            mainDataFrame = pd.DataFrame(
                columns=['dataid', 'localminute', 'car1', 'car2', 'grid', 'solar', 'solar2', 'leg1v', 'leg2v'])
            for chunk in tqdm(pd.read_csv(file_path, chunksize=100000,
                                          usecols=['dataid', 'localminute', 'car1', 'car2', 'grid', 'solar', 'solar2',
                                                   'leg1v', 'leg2v'])):
                df = chunk[chunk['dataid'] == id]
                mainDataFrame = mainDataFrame.append(df)

            mainDataFrame.to_csv(output_path + r"\DataID_" + str(id) + '_1_second.csv', index=False)
            print(str(id) + ' is done')

    @staticmethod
    def dataPreprocessor(dataFrame):
        dataFrame['localminute'] = pd.to_datetime(dataFrame['localminute'], format='%Y-%m-%d %H:%M:%S-%f')
        dataFrame = dataFrame.sort_values(by='localminute')
        dataFrame['total_power'] = dataFrame['grid'] + dataFrame['solar']
        dataFrame['total_power'] = dataFrame['total_power'].replace(np.NAN, 0, regex=True)
        dataFrame['car1'] = dataFrame['car1'].replace(np.NAN, 0, regex=True)
        dataFrame['EV_label'] = [1 if int(i) >= 1 else 0 for i in list(dataFrame['car1'])]
        dataFrame['EV_label'] = dataFrame['EV_label'].replace(np.NAN, 0, regex=True)
        dataFrame = dataFrame[['localminute', 'dataid', 'total_power', 'EV_label']]
        return dataFrame

    def combineDataIDs(self, input_folder, output_csv_path):
        csvList = glob(input_folder + "\*.csv")
        mainDataFrame = pd.DataFrame(columns=['localminute', 'dataid', 'total_power', 'EV_label'])
        for csv in csvList:
            print(csv)
            dataFrame = pd.read_csv(csv)
            processedDataFrame = self.dataPreprocessor(dataFrame)
            mainDataFrame = mainDataFrame.append(processedDataFrame)

        mainDataFrame.to_csv(output_csv_path)


class CustomCallback(callbacks.Callback):
    """ This is to log the Epochs in the Keras model iterations a callback"""

    def on_train_begin(self, logs=None):
        logging.info("Training Started")
        pass

    def on_train_end(self, logs=None):
        logging.info("Training Finished")
        pass

    def on_epoch_begin(self, epoch, logs=None):
        logging.info("epoch number " + str(epoch) + " started")

    def on_epoch_end(self, epoch, logs=None):
        logging.info("epoch number " + str(epoch) + " finished")


class DeepLearning:
    X = None
    Y = None
    trainX, testX = None, None
    trainY, testY = None, None
    xScaler, yScaler = None, None
    scaled_X_train, scaled_y_train = None, None
    n_input = None  # how many samples/rows/timesteps to look in the past in order to forecast the next sample
    batchSize = None  # Number of time series samples in each batch
    epochs = None
    model = None
    modelName = None
    dataset = None
    folderName = None
    accuracyReport = None
    history = None
    checkpoint = None
    rpath = None
    epochNumber = None

    def __init__(self):

        pass

    def restVariables(self):
        self.X = None
        self.Y = None
        self.trainX, self.testX = None, None
        self.trainY, self.testY = None, None
        self.xScaler, self.yScaler = None, None
        self.scaled_X_train, self.scaled_y_train = None, None
        self.n_input = None
        self.batchSize = None
        self.epochs = None
        self.model = None
        self.modelName = None
        self.dataset = None
        self.folderName = None
        self.accuracyReport = None
        self.history = None

    def __del__(self):
        pass

    def runModel(self, csv_path, n_input, batchSize, epochs, modelName, dataset, loadFromCheckPoint=None,
                 rpath=r"Data_iteration/"):
        self.n_input = n_input
        self.batchSize = batchSize
        self.epochs = epochs
        self.epochNumber = epochs
        self.rpath = rpath
        self.prepareData(csv_path)
        self.modelName = modelName
        self.dataset = dataset
        self.createFolder()  # create folder containing results
        if loadFromCheckPoint:
            last_epoch = int(loadFromCheckPoint.split('.')[-2])
            self.epochNumber = self.epochNumber + last_epoch
            self.model = load_model(loadFromCheckPoint)
            self.model.summary()
            print("Last saved Checkpoint loaded successfully")
            generator = TimeseriesGenerator(self.scaled_X_train, self.scaled_y_train, length=self.n_input,
                                            batch_size=self.batchSize)
            self.history = self.model.fit_generator(generator, epochs=self.epochs,
                                                    callbacks=[CustomCallback(), self.checkpoint])
        else:
            if modelName == 'LSTM1':
                self.LSTM1()
            elif modelName == 'LSTM2':
                self.LSTM2()
            elif modelName == 'LSTM3':
                self.LSTM3()
            elif modelName == 'GRU':
                self.GRU()

        self.evaluation()
        self.trainingValidation()
        self.save_model()
        logging.info("Training Successfully Finished")
        logging.shutdown()
        self.restVariables()

    def createFolder(self):
        today = datetime.now()
        self.folderName = today.strftime('%d_%m_%Y__%H_%M')
        if not os.path.exists(self.rpath + self.folderName):
            os.makedirs(self.rpath + self.folderName)

        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=self.rpath + self.folderName + "/IterationLog.log", level=logging.INFO)
        logging.info('Log File is Created Successfully')

        if not os.path.exists(self.rpath + self.folderName + "/checkpoints"):
            os.makedirs(self.rpath + self.folderName + "/checkpoints")

        self.checkpoint = callbacks.ModelCheckpoint(self.rpath + self.folderName + "/checkpoints/" + "{epoch:02d}.hdf5",
                                                    monitor='loss', verbose=1, save_best_only=False, mode='auto',
                                                    period=1)

    def prepareData(self, csv_path):
        sourceData = pd.read_csv(csv_path)

        sourceData['localminute'] = pd.to_datetime(sourceData['localminute'], format='%Y-%m-%d %H:%M:%S.%f')
        sourceData = sourceData.set_index('localminute')
        sourceData = sourceData.resample('1T').first()
        print("========= resample to 1 min is done============")
        self.X = sourceData[['total_power']]  # independent Variable
        self.Y = sourceData[['EV_label']]  # target variable

        # split the data into train and test
        self.trainX, self.testX = train_test_split(self.X, test_size=0.3, shuffle=False)
        self.trainY, self.testY = train_test_split(self.Y, test_size=0.3, shuffle=False)

        print("Shape of TrainX and TrainY ", self.trainX.shape, self.trainY.shape)
        print("Shape of TestX and TestY ", self.trainY.shape, self.testY.shape)

        self.xScaler = MinMaxScaler(feature_range=(0, 1))  # scale so that all the X data will range from 0 to 1
        self.xScaler.fit(self.trainX)
        self.scaled_X_train = self.xScaler.transform(self.trainX)
        print('Scaled Train X Shape ', self.trainX.shape)

        self.yScaler = MinMaxScaler(feature_range=(0, 1))
        self.yScaler.fit(self.trainY)
        self.scaled_y_train = self.trainY.to_numpy()
        self.scaled_y_train = self.scaled_y_train.reshape(
            -1)  # remove the second dimension from y so the shape changes from (n,1) to (n,)
        self.scaled_y_train = np.insert(self.scaled_y_train, 0, 0)
        self.scaled_y_train = np.delete(self.scaled_y_train, -1)
        print('Scaled Train Y Shape ', self.scaled_y_train.shape)

    def LSTM1(self):
        n_features = self.trainX.shape[1]  # how many predictors/Xs/features we have to predict y
        generator = TimeseriesGenerator(self.scaled_X_train, self.scaled_y_train, length=self.n_input,
                                        batch_size=self.batchSize)

        #  ================= Keras Model LSTM Build ===========================
        self.model = Sequential()
        self.model.add(LSTM(150, activation='relu', input_shape=(self.n_input, n_features)))
        self.model.add(
            Dense(1, activation='sigmoid'))  # since it is a binary classification, we are calling sigmoid function
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

        self.history = self.model.fit_generator(generator, epochs=self.epochs,
                                                callbacks=[CustomCallback(), self.checkpoint])

    def LSTM2(self):
        n_features = self.trainX.shape[1]  # how many predictors/Xs/features we have to predict y
        generator = TimeseriesGenerator(self.scaled_X_train, self.scaled_y_train, length=self.n_input,
                                        batch_size=self.batchSize)

        #  ================= Keras Model LSTM Build ===========================
        self.model = Sequential()
        self.model.add(LSTM(150, activation='sigmoid', input_shape=(self.n_input, n_features)))
        self.model.add(Dropout(0.5))
        self.model.add(
            Dense(1, activation='sigmoid'))  # since it is a binary classification, we are calling sigmoid function
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

        self.history = self.model.fit_generator(generator, epochs=self.epochs,
                                                callbacks=[CustomCallback(), self.checkpoint])

    def LSTM3(self):
        n_features = self.trainX.shape[1]  # how many predictors/Xs/features we have to predict y
        generator = TimeseriesGenerator(self.scaled_X_train, self.scaled_y_train, length=self.n_input,
                                        batch_size=self.batchSize)

        #  ================= Keras Model LSTM Build ===========================
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.n_input, n_features)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(
            Dense(1, activation='sigmoid'))  # since it is a binary classification, we are calling sigmoid function
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
        self.history = self.model.fit_generator(generator, epochs=self.epochs,
                                                callbacks=[CustomCallback(), self.checkpoint])

    def GRU(self):
        n_features = self.trainX.shape[1]  # how many predictors/Xs/features we have to predict y
        generator = TimeseriesGenerator(self.scaled_X_train, self.scaled_y_train, length=self.n_input,
                                        batch_size=self.batchSize)

        #  ================= Keras Model GRU Build ===========================
        self.model = Sequential()
        self.model.add(GRU(50, return_sequences=True, input_shape=(self.n_input, n_features)))
        self.model.add(Dropout(0.2))
        self.model.add(GRU(100, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(
            Dense(1, activation='sigmoid'))  # since it is a binary classification, we are calling sigmoid function
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

        self.history = self.model.fit_generator(generator, epochs=self.epochs,
                                                callbacks=[CustomCallback(), self.checkpoint])

    def evaluation(self):

        scaled_X_test = self.xScaler.transform(self.testX)
        test_generator = TimeseriesGenerator(scaled_X_test, np.zeros(len(self.testX)), length=self.n_input,
                                             batch_size=self.batchSize)

        y_pred_scaled = self.model.predict(test_generator)
        y_pred = self.yScaler.inverse_transform(y_pred_scaled)
        y_pred = y_pred.ravel().tolist()

        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        y_true = self.testY.values[self.n_input:].ravel().tolist()
        self.accuracyReport = classification_report(y_true, y_pred)
        logging.info(self.accuracyReport)
        y_true = [-0.25 if i == 1 else -0.5 for i in y_true]
        y_pred = [-0.75 if i >= 0.5 else -1 for i in y_pred]

        results = pd.DataFrame({'yTrue': y_true, 'yPred': y_pred})
        results.plot(title="Model Name: " + self.modelName + ", No.of Epoch: " + str(self.epochNumber)
                           + ", Batch Size: " + str(self.batchSize) + ", n_inputs: " + str(self.n_input))
        x = self.testX['total_power']
        normalized = (x - min(x)) / (max(x) - min(x))
        plt.plot(list(normalized))
        plt.legend(['Y True', 'Y Predicted', 'Aggregated Power (normalized)'], loc="upper right")

        figure = plt.gcf()  # get current figure
        figure.set_size_inches(12, 6)
        # when saving, specify the DPI
        plt.savefig(self.rpath + self.folderName + "/Result.png", dpi=100)
        plt.close(figure)
        logging.info("Exported Results Successfully")

    def trainingValidation(self):
        loss = self.history.history['loss']
        epochs = self.history.epoch
        logging.info("Loss values: " + str(list(loss)))
        plt.plot(epochs, loss)
        plt.title('model train loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(12, 6)
        # when saving, specify the DPI
        plt.savefig(self.rpath + self.folderName + "/training.png", dpi=100)
        plt.close(figure)

    def save_model(self):
        if not os.path.exists(self.rpath + self.folderName + "/model"):
            os.makedirs(self.rpath + self.folderName + "/model")

        model_json = self.model.to_json()
        with open(self.rpath + self.folderName + "/model/" + "model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(self.rpath + self.folderName + "/model/" + "model.h5")
        logging.info("Saved model to disk")

    def predictFromTheSavedModel(self, json_path, weights_path, data):
        from keras.models import model_from_json
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(weights_path)

        print("Loaded model from disk")
        sourceData = pd.read_csv(data)
        sourceData = sourceData.replace(np.NAN, 0, regex=True)
        sourceData['total_power'] = sourceData['total_power'].apply(lambda x: x if x > 0 else 0)
        self.X = sourceData[['total_power']]  # independent Variable
        self.Y = sourceData[['EV_label']]  # target variable

        self.xScaler = MinMaxScaler(feature_range=(0, 1))  # scale so that all the X data will range from 0 to 1
        self.xScaler.fit(self.X)
        self.scaled_X_train = self.xScaler.transform(self.X)

        self.yScaler = MinMaxScaler(feature_range=(0, 1))
        self.yScaler.fit(self.Y)
        self.scaled_y_train = self.Y.to_numpy()
        self.scaled_y_train = self.scaled_y_train.reshape(
            -1)  # remove the second dimension from y so the shape changes from (n,1) to (n,)
        self.scaled_y_train = np.insert(self.scaled_y_train, 0, 0)
        self.scaled_y_train = np.delete(self.scaled_y_train, -1)

        test_generator = TimeseriesGenerator(self.scaled_X_train, np.zeros(len(self.X)), length=25,
                                             batch_size=60)

        y_pred_scaled = self.model.predict(test_generator)
        y_pred = self.yScaler.inverse_transform(y_pred_scaled)
        y_pred = y_pred.ravel().tolist()

        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        y_true = self.Y.values[25:].ravel().tolist()
        self.accuracyReport = classification_report(y_true, y_pred)
        logging.info(self.accuracyReport)
        y_true = [-0.25 if i == 1 else -0.5 for i in y_true]
        y_pred = [-0.75 if i >= 0.5 else -1 for i in y_pred]

        results = pd.DataFrame({'yTrue': y_true, 'yPred': y_pred})
        results.plot(title="prediction on the saved model")
        x = self.X['total_power']
        normalized = (x - min(x)) / (max(x) - min(x))
        plt.plot(list(normalized))
        plt.legend(['Y True', 'Y Predicted', 'Aggregated Power (normalized)'], loc="upper right")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(12, 6)
        # when saving, specify the DPI
        plt.savefig("result.png", dpi=100)
        plt.close(figure)

        print(self.accuracyReport)
