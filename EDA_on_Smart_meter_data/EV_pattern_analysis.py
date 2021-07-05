from matplotlib import pyplot as plt
import pandas as pd
# import seaborn as sns


data_path = r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\1s_Austine_data\DataID_4767_1_second.csv"

sourceData = pd.read_csv(data_path)


sourceData['localminute'] = pd.to_datetime(sourceData['localminute'], format='%Y-%m-%d %H:%M:%S-%f')
sourceData = sourceData.sort_values(by='localminute')

# ========== AC1 plot ==================
car1 = sourceData['car1']
timestamp = sourceData['localminute']
plt.title('EV Power vs Timestamp - Austine Data id 4767')
plt.xlabel('Time Stamp')
plt.ylabel('Power (Kw)')
plt.plot(timestamp,car1)
