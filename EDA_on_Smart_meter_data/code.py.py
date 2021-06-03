from matplotlib import pyplot as plt
import pandas as pd
# import seaborn as sns


data_path = r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CONVERTED DATA\floor1.csv"

sourceData = pd.read_csv(data_path)

sourceData[["Day", "Month"]] = sourceData["Date"].str.split(pat=":", expand=True)
sourceData['time_stamp'] = pd.to_datetime(sourceData[['Year', 'Month', 'Day']]) + pd.to_timedelta(sourceData['Time'])

# ========== AC1 plot ==================
ac1 = sourceData['AC2']#[40000:65000]
time_stamp =sourceData['time_stamp']#[40000:65000]
plt.plot(time_stamp, ac1)

# # ========== AC1 plot wrt watts recived per hour==================
# # ac1 = sourceData['wh_recieved'][40000:65000]
# # time_stamp =sourceData['time_stamp'][40000:65000]
# # plt.title('wh_recieved')
# # plt.plot(time_stamp, ac1) # we can understand that watts received is the cumulative sum of every every second
#
#
#
# # ========== AC1 plot wrt watts total ==================
# # x = sourceData['watts_total'][40000:65000]
# # normalized = (x-min(x))/(max(x)-min(x))
# # time_stamp =sourceData['time_stamp'][40000:65000]
# # plt.title('AC1 plot wrt watts total')
# # plt.plot(time_stamp, normalized)
#
#
# # ========== AC1 plot wrt r watts total ==================
# x = sourceData['watts_r'][40000:65000]
# normalized = (x-min(x))/(max(x)-min(x))
# time_stamp =sourceData['time_stamp'][40000:65000]
# plt.title('AC1 plot wrt watts_r')
# plt.plot(time_stamp, normalized)
#
#
# # ========== AC1 voltage & Current graph in R channel ================
# volt_ampere = sourceData['v_r'][40000:45000]
# current = sourceData['I_r'][40000:45000]
# plt.title('AC1 voltage & Current graph in R channel- plot')
# plt.ylabel('Current'),plt.xlabel('voltage')
# plt.plot(volt_ampere, current)
#


# ========== AC1  & Current graph in R channel ================

x = sourceData['I_y']#[40000:65000]
normalized = (x-min(x))/(max(x)-min(x))
time_stamp =sourceData['time_stamp']#[40000:65000]
plt.title('AC1 Current graph in R channel- plot')
plt.ylabel('Current'),plt.xlabel('voltage')
plt.plot(time_stamp,normalized)


# # ============ Correlation Heat Map ==================
# sourceData_dropped = sourceData.drop(sourceData.columns[0:19], axis=1)
# sourceData_dropped = sourceData_dropped.drop(['var_tot','var_r','var_y','var_b','frequency']
#                                              , axis=1)
# plt.title('Correlation Heat Map ')
# sns.heatmap(sourceData_dropped.corr())

