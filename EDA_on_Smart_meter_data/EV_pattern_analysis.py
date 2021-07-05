from matplotlib import pyplot as plt
import pandas as pd
# import seaborn as sns


data_path = r"C:\Users\sivashankar.palraj\PycharmProjects\NY\DataID_1222_1_second.csv"

sourceData = pd.read_csv(data_path)
sourceData = sourceData.set_index('localminute')

sourceData.index = pd.to_datetime(sourceData.index, format='%Y-%m-%d %H:%M:%S-%f')
sourceData = sourceData.sort_index().loc['2019-05-01 00:00:00':'2019-05-01 01:00:00']

# ========== AC1 plot ==================
car1 = sourceData['car1']
timestamp = sourceData.index#.loc['2019-05-01':'2019-05-02']
plt.title('EV Power vs Timestamp - NY Data id 1222')
plt.xlabel('Time Stamp')
plt.ylabel('Power (Kw)')
plt.plot(timestamp,car1)
