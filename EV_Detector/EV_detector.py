from tqdm import tqdm
from glob import glob
import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras import callbacks
from keras.layers import LSTM, Dense, Dropout, GRU, AveragePooling1D
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import classification_report
import os
from datetime import datetime
import logging
import tensorflow as tf
from keras.models import model_from_json

json_path = r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\EV_Detector\model\model.json"
weights_path = r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\EV_Detector\model\model.h5"
data = r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\EV_Detector\data\Combined_50_houses_austine - Copy.csv"

json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(weights_path)

print("Loaded model from disk")
sourceData = pd.read_csv(data)
sourceData = sourceData.replace(np.NAN, 0, regex=True)
X = sourceData[['total_power']]  # independent Variable


xScaler = MinMaxScaler(feature_range=(0, 1))  # scale so that all the X data will range from 0 to 1
xScaler.fit(X)
scaled_X_train = xScaler.transform(X)

test_generator = TimeseriesGenerator(scaled_X_train, np.zeros(len(X)), length=60,
                                     batch_size=100)

yScaler = MinMaxScaler(feature_range=(0, 1))
yScaler.fit(pd.DataFrame({'label': [0,1]}))

y_pred_scaled = model.predict(test_generator)
y_preds = yScaler.inverse_transform(y_pred_scaled)
y_preds = y_preds.ravel().tolist()

y_pred = [1 if i >= 0.5 else 0 for i in y_preds]


def convertSecToDay(n):
    day = n // (24 * 3600)
    n = n % (24 * 3600)
    hour = n // 3600
    n %= 3600
    minutes = n // 60
    n %= 60
    seconds = n
    return {'days': str(day),
            'hours': str(hour),
            'minutes': str(minutes),
            'seconds': str(seconds)}


evContrib_index = []
evContrib = []
count = 0
for i, j in zip(X['total_power'], y_pred):
    if j == 1:
        evContrib.append(i)
        evContrib_index.append(count)
    count += 1

maximumEVPower = max(evContrib)
minimumEVPower = min(evContrib)

totalDuration = convertSecToDay(len(sourceData))
EV_contribution = convertSecToDay(y_pred.count(1))
EVPercentageOnTotalPower = (y_pred.count(1) / len(sourceData)) * 100

plt.plot(X)
plt.title('Aggregated Power vs EV Contribution')
plt.xlabel('Time Stamp')
plt.ylabel('Power (Kw)')
plt.plot(evContrib_index, evContrib, color="red")
plt.legend(['Aggregated Power', 'EV Charging Event'], loc="upper right")
figure = plt.gcf()  # get current figure
figure.set_size_inches(12, 6)
# when saving, specify the DPI
plt.savefig("result.png", dpi=100)
plt.close(figure)

from fpdf import FPDF

pdf = FPDF('P', 'mm', 'A4')
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(80, 10, 'Electric Vehicle Detection Report - Postmortem Analysis', 'C')
pdf.ln()
pdf.set_font('Arial', 'B', 8)
pdf.cell(80, 10, 'Total Duration: ' + totalDuration['days'] + " Days " + totalDuration['hours'] + " hours " +
         totalDuration['minutes'] + " minutes " + totalDuration['seconds'] + " seconds")
pdf.ln()
pdf.cell(80, 10,
         'Total EV Charging Duration: ' + EV_contribution['days'] + " Days " + EV_contribution['hours'] + " hours " +
         EV_contribution['minutes'] + " minutes " + EV_contribution['seconds'] + " seconds")
pdf.ln()
pdf.cell(80, 10, 'EV Charging Minimum Aggregated Power: ' + str(round(minimumEVPower,2)) + " kW")
pdf.ln()
pdf.cell(80, 10, 'EV Charging Maximum Aggregated Power: ' + str(round(maximumEVPower,2))+ " kW")
pdf.ln()
pdf.cell(80, 10, 'EV Charging vs Total Duration: ' + str(round(EVPercentageOnTotalPower, 2)) + " %")
pdf.ln()
pdf.image('result.png', x=0, y=None, w=220, h=110, type='', link='')
pdf.output(r'report/EV Report.pdf', 'F')
