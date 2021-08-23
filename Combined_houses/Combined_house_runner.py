from tqdm import tqdm
from glob import glob
import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
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

        dataIds = newyorkIds

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
    def dataPreprocessor(dataFrame,start,end):
        dataFrame['localminute'] = pd.to_datetime(dataFrame['localminute'], format='%Y-%m-%d %H:%M:%S-%f')
        dataFrame = dataFrame.sort_values(by='localminute')
        dataFrame['total_power'] = dataFrame['grid'] + dataFrame['solar']
        dataFrame['total_power'] = dataFrame['total_power'].replace(np.NAN, 0, regex=True)
        dataFrame['car1'] = dataFrame['car1'].replace(np.NAN, 0, regex=True)
        dataFrame['EV_label'] = [1 if i >= 0.3 else 0 for i in list(dataFrame['car1'])]
        dataFrame['EV_label'] = dataFrame['EV_label'].replace(np.NAN, 0, regex=True)
        dataFrame = dataFrame[['localminute', 'total_power', 'EV_label']]
        mask = (dataFrame['localminute'] > start) & (
                dataFrame['localminute'] <= end)
        dataFrame = dataFrame.loc[mask]
        dataFrame.set_index('localminute', inplace=True)
        dataFrame = dataFrame.resample('1s').first()
        total_power = dataFrame['total_power'].interpolate()
        EV_label = dataFrame['EV_label'].interpolate()
        dataFrame = pd.DataFrame({'total_power': total_power, 'EV_label': EV_label})
        dataFrame = dataFrame.reset_index()
        return dataFrame

    def mergeDataIDs(self, input_folder, output_csv_path):
        csvList = glob(input_folder + "\*.csv")
        localminute = []
        total_power = []
        EV_label = []
        initial = True
        for csv in csvList:
            print(csv)
            dataFrame = pd.read_csv(csv)
            if '_zz' in csv:
                processedDataFrame = self.dataPreprocessor(dataFrame,"2018-04-01 00:00:00","2018-06-30 00:00:00")
            else:
                processedDataFrame = self.dataPreprocessor(dataFrame, "2019-05-01 00:00:00", "2019-07-30 00:00:00")
            if initial:
                localminute = processedDataFrame['localminute']
                total_power = processedDataFrame['total_power']
                EV_label = processedDataFrame['EV_label']
                initial = False
            else:
                total_power = [i + j for i, j in zip(total_power, processedDataFrame['total_power'])]
                EV_label = [0 if i + j == 0 else 1 for i, j in zip(EV_label, processedDataFrame['EV_label'])]

        mainDataFrame = pd.DataFrame({'localminute': localminute,
                                      'total_power': total_power,
                                      'EV_label': EV_label})

        mainDataFrame.to_csv(output_csv_path)


dl = DataReader()
dl.mergeDataIDs( r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\comb",
                r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\comb\combined_50_houses_austine.csv")