import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from glob import glob


class DataReader:

    def __init__(self):

        pass

    @staticmethod
    def dataExtractorOneSecond(file_path, output_path):

        austineIds = [6139, 3039, 3538, 8386, 4031, 9922, 7951, 8565, 9278, 661, 7800, 9160, 8156, 7536, 2361, 2818,
                      4767, 3456, 9019, 7901, 7719, 5746, 1642, 4373, 2335]

        for id in austineIds:
            source_dict = {'dataid': [],
                           'localminute': [],
                           'car1': [],
                           'car2': [],
                           'grid': [],
                           'solar': [],
                           'solar2': [],
                           'leg1v': [],
                           'leg2v': [],
                           }
            for chunk in tqdm(pd.read_csv(file_path, chunksize=100000)):
                df = chunk[chunk['dataid'] == id]
                df = df[['dataid', 'localminute', 'car1', 'car2', 'grid', 'solar', 'solar2', 'leg1v', 'leg2v']]
                for i, j in df.iterrows():
                    for k, l in j.items():
                        source_dict[k].append(l)

            out_df = pd.DataFrame(source_dict)
            out_df.to_csv(output_path + r"\DataID_" + str(id) + '_1_second.csv')
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

    def __init__(self):
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
        pass

    def prepareData(self, csv_path):
        sourceData = pd.read_csv(csv_path)
        self.X = sourceData[['current']]  # independent Variable
        self.Y = sourceData[['EV_label']]  # target variable

        # split the data into train and test
        self.trainX, self.testX = train_test_split(X, test_size=0.3, shuffle=False)
        self.trainY, self.testY = train_test_split(Y, test_size=0.3, shuffle=False)

        print("Shape of TrainX and TrainY ", self.trainX.shape, self.trainY.shape)
        print("Shape of TestX and TestY ", self.trainY.shape, self.testY.shape)

        self.Xscaler = MinMaxScaler(feature_range=(0, 1))  # scale so that all the X data will range from 0 to 1
        self.Xscaler.fit(trainX)
        self.scaled_X_train = Xscaler.transform(self.trainX)
        print('Scaled Train X Shape ', self.trainX.shape)

        self.Yscaler = MinMaxScaler(feature_range=(0, 1))
        self.Yscaler.fit(trainY)
        self.scaled_y_train = trainY.to_numpy()
        self.scaled_y_train = scaled_y_train.reshape(
            -1)  # remove the second dimension from y so the shape changes from (n,1) to (n,)
        self.scaled_y_train = np.insert(scaled_y_train, 0, 0)
        self.scaled_y_train = np.delete(scaled_y_train, -1)
        print('Scaled Train Y Shape ', self.scaled_y_train.shape)

    def LSTM(self):
        n_features = self.trainX.shape[1]  # how many predictors/Xs/features we have to predict y
        generator = TimeseriesGenerator(self.scaled_X_train, self.scaled_y_train, length=self.n_input,
                                        batch_size=self.batchSize)

        #  ================= Keras Model LSTM Build ===========================
        model = Sequential()
        model.add(LSTM(150, activation='relu', input_shape=(self.n_input, n_features)))
        model.add(
            Dense(1, activation='sigmoid'))  # since it is a binary classification, we are calling sigmoid function
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit_generator(generator, epochs=self.epochs)
