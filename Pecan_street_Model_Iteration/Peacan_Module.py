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


# =============== To Extract the one second Data based on IDs ====================================

pecan = DataReader()
pecan.dataExtractorOneSecond(
    r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\file2.csv",
    r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\1s_Austine_data")


# =============== Convert Extracted data's to single csv with preprocessed ====================================

pecan.combineDataIDs(r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\1s_Austine_data",
                    r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\tain_data\Combined_CSV.csv")

