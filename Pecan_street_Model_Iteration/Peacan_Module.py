import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
from keras import callbacks
from keras.layers import LSTM, Dense
from matplotlib import pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import classification_report
import os
from datetime import datetime
import logging


class DataReader:

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

    def dataPreprocessor(self, dataFrame):
        dataFrame['localminute'] = pd.to_datetime(dataFrame['localminute'], format='%Y-%m-%d %H:%M:%S-%f')
        dataFrame = dataFrame.sort_values(by='localminute')
        dataFrame['total_power'] = dataFrame['grid'] + dataFrame['solar']
        dataFrame['car1'] = dataFrame['car1'].replace(np.NAN, 0, regex=True)
        dataFrame['EV_label'] = [1 if int(i) >= 1 else 0 for i in list(dataFrame['car1'])]
        dataFrame = dataFrame[['localminute', 'dataid', 'total_power', 'EV_label']]
        return dataFrame

    def combineDataIDs(self, input_folder, output_csv_path):
        csvList = glob(input_folder + "\*.csv")
        mainDataFrame = pd.DataFrame(columns=['localminute', 'dataid', 'total_power', 'EV_label'])
        for csv in csvList:
            dataFrame = pd.read_csv(csv)
            processedDataFrame = self.dataPreprocessor(dataFrame)
            mainDataFrame = mainDataFrame.append(processedDataFrame)

        mainDataFrame.to_csv(output_csv_path)


class CustomCallback(callbacks.Callback):
    def on_train_begin(self, logs=None):
        logging.info("Training Started")
        pass

    def on_train_end(self, logs=None):
        logging.info("Training Finished")
        pass

    def on_epoch_begin(self, epoch, logs=None):
        logging.info("epoch number "+str(epoch) +" started")

    def on_epoch_end(self, epoch, logs=None):
        logging.info("epoch number "+str(epoch)+" finished")


class DeepLearning:
    X = None
    Y = None
    trainX, testX = None, None
    trainY, testY = None, None
    Xscaler, Yscaler = None, None
    scaled_X_train, scaled_y_train = None, None
    n_input = None  # how many samples/rows/timesteps to look in the past in order to forecast the next sample
    batchSize = None  # Number of time series samples in each batch
    epochs = None
    model = None
    modelName = None
    dataset = None
    folderName = None
    accuracyReport = None

    def __init__(self):
        pass

    def runModel(self, csv_path, n_input, batchSize, epochs, modelName, dataset):
        self.n_input = n_input
        self.batchSize = batchSize
        self.epochs = epochs
        self.prepareData(csv_path)
        self.modelName = modelName
        self.dataset = dataset
        self.createFolder()  # create folder containing results

        if modelName == 'LSTM1':
            self.LSTM1()
        elif modelName == 'LSTM2':
            self.LSTM2()
        self.evaluation()
        logging.shutdown()

    def createFolder(self):
        today = datetime.now()
        self.folderName = today.strftime('%d_%m_%Y__%H_%M')
        if not os.path.exists(r"Data_iteration/" + self.folderName):
            os.makedirs(r"Data_iteration/" + self.folderName)

        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=r"Data_iteration/" + self.folderName + r"\IterationLog.log", level=logging.INFO)
        logging.info('Log File is Created Successfully')

    def prepareData(self, csv_path):
        sourceData = pd.read_csv(csv_path)
        self.X = sourceData[['total_power']]  # independent Variable
        self.Y = sourceData[['EV_label']]  # target variable

        # split the data into train and test
        self.trainX, self.testX = train_test_split(self.X, test_size=0.3, shuffle=False)
        self.trainY, self.testY = train_test_split(self.Y, test_size=0.3, shuffle=False)

        print("Shape of TrainX and TrainY ", self.trainX.shape, self.trainY.shape)
        print("Shape of TestX and TestY ", self.trainY.shape, self.testY.shape)

        self.Xscaler = MinMaxScaler(feature_range=(0, 1))  # scale so that all the X data will range from 0 to 1
        self.Xscaler.fit(self.trainX)
        self.scaled_X_train = self.Xscaler.transform(self.trainX)
        print('Scaled Train X Shape ', self.trainX.shape)

        self.Yscaler = MinMaxScaler(feature_range=(0, 1))
        self.Yscaler.fit(self.trainY)
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

        self.model.fit_generator(generator, epochs=self.epochs, callbacks=[CustomCallback()])

    def LSTM2(self):
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

        self.model.fit_generator(generator, epochs=self.epochs, callbacks=[CustomCallback()])


    def evaluation(self):

        scaled_X_test = self.Xscaler.transform(self.testX)
        test_generator = TimeseriesGenerator(scaled_X_test, np.zeros(len(self.testX)), length=self.n_input,
                                             batch_size=self.batchSize)

        y_pred_scaled = self.model.predict(test_generator)
        y_pred = self.Yscaler.inverse_transform(y_pred_scaled)
        y_pred = y_pred.ravel().tolist()

        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        y_true = self.testY.values[self.n_input:].ravel().tolist()
        self.accuracyReport = classification_report(y_true, y_pred)
        logging.info(self.accuracyReport)
        y_true = [-0.25 if i == 1 else -0.5 for i in y_true]
        y_pred = [-0.75 if i >= 0.5 else -1 for i in y_pred]

        results = pd.DataFrame({'yTrue': y_true, 'yPred': y_pred})
        results.plot(title="Model Name: " + self.modelName + ", No.of Epoch: " + str(self.epochs)
                           + ", Batch Size: " + str(self.batchSize) + ", n_inputs: " + str(self.n_input))
        x = self.testX['total_power']
        normalized = (x - min(x)) / (max(x) - min(x))
        plt.plot(list(normalized))
        plt.legend(['Y True', 'Y Predicted', 'Aggregated Power (normalized)'], loc="upper right")

        figure = plt.gcf()  # get current figure
        figure.set_size_inches(12, 6)
        # when saving, specify the DPI
        plt.savefig(r"Data_iteration/" + self.folderName + "/Result.png", dpi=500)
        plt.close()
        logging.info("Exported Results Successfully")