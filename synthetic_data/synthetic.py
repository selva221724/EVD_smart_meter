import pandas as pd
import numpy as np
import datetime
from glob import glob


class SyntheticDataCreator:
    pecanDataset = None
    SMDateset = None
    mergedDateset = None

    def __init__(self):
        pass

    def dataSynthesis(self, pecandata, p_timeFrame, SMData, s_timeFrame):
        self.PecanProcessor(pecandata, p_timeFrame)
        self.SMDProcessor(SMData, s_timeFrame)
        self.dataMerger()
        return self.mergedDateset

    def datacombiner(self,path,out_path):
        csvList = glob(path + "\*.csv")
        mainDataFrame = pd.DataFrame(columns=['index','localminute','car1','EV_label','watts_total','total_power'])
        for csv in csvList:
            print(csv)
            dataFrame = pd.read_csv(csv)
            mainDataFrame = mainDataFrame.append(dataFrame)

        mainDataFrame.to_csv(out_path)

    def PecanProcessor(self, data, timeFrame):
        sourceData = pd.read_csv(data)
        sourceData['localminute'] = pd.to_datetime(sourceData['localminute'], format='%Y-%m-%d %H:%M:%S-%f')
        sourceData = sourceData.sort_values(by='localminute')
        sourceData['car1'] = sourceData['car1'].replace(np.NAN, 0, regex=True)
        sourceData['EV_label'] = [1 if int(i) >= 1 else 0 for i in list(sourceData['car1'])]
        sourceData['EV_label'] = sourceData['EV_label'].replace(np.NAN, 0, regex=True)
        sourceData = sourceData[['localminute', 'car1', 'EV_label']]
        mask = (sourceData['localminute'] > timeFrame['start']) & (
                sourceData['localminute'] <= timeFrame['end'])
        sourceData = sourceData.loc[mask]
        self.pecanDataset = sourceData

    def SMDProcessor(self, data, timeFrame):
        sourceData = pd.read_csv(data)
        sourceData[["Day", "Month"]] = sourceData["Date"].str.split(pat=":", expand=True)
        sourceData['localminute'] = pd.to_datetime(sourceData[['Year', 'Month', 'Day']]) + pd.to_timedelta(
            sourceData['Time'])
        sourceData = sourceData.sort_values(by='localminute')
        sourceData['watts_total'] = sourceData['watts_total'].apply(lambda x:float(x.split('(')[1].split(",")[0]))
        sourceData = sourceData[['localminute', 'watts_total']]
        mask = (sourceData['localminute'] > timeFrame['start']) & (
                sourceData['localminute'] <= timeFrame['end'])
        sourceData = sourceData.loc[mask]
        self.SMDateset = sourceData

    def dataMerger(self):
        #
        #
        # self.pecanDataset['localminute'] = pd.to_datetime(self.pecanDataset['localminute'])
        # self.pecanDataset.set_index('localminute', inplace=True)
        # self.pecanDataset = self.pecanDataset.resample('1s').first()
        # self.pecanDataset = self.pecanDataset['car1','EV_label'].interpolate()
        # self.pecanDataset = pd.DataFrame(self.pecanDataset)
        self.pecanDataset = self.pecanDataset.reset_index()

        self.SMDateset['localminute'] = pd.to_datetime(self.SMDateset['localminute'])
        self.SMDateset.set_index('localminute', inplace=True)
        self.SMDateset = self.SMDateset.resample('1s').first()
        self.SMDateset = self.SMDateset['watts_total'].interpolate()
        self.SMDateset = pd.DataFrame(self.SMDateset)
        self.SMDateset = self.SMDateset.reset_index()

        self.mergedDateset = self.pecanDataset
        self.mergedDateset['watts_total'] = self.SMDateset['watts_total']
        self.mergedDateset['total_power'] = (self.mergedDateset['watts_total'] / 1000) + self.mergedDateset['car1']
