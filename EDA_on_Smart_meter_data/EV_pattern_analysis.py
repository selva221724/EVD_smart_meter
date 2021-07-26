from matplotlib import pyplot as plt
import pandas as pd

# plt.switch_backend('Agg')
# import seaborn as sns


data_path = r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\1s_Austine_data\DataID_661_1_second.csv"

sourceData = pd.read_csv(data_path)

sourceData['localminute'] = pd.to_datetime(sourceData['localminute'], format='%Y-%m-%d %H:%M:%S-%f')
sourceData = sourceData.sort_values(by='localminute')

# ========== AC1 plot ==================
car1 = sourceData['car1']
timestamp = sourceData['localminute']
plt.title('EV Power vs Timestamp - Austine Data id 4767')
plt.xlabel('Time Stamp')
plt.ylabel('Power (Kw)')
plt.plot(timestamp, car1)
#
# major_id = 1
# car = None
# time = None
# count = 0
# start = None
# end = None
# mid_count = 0
# for i in car1:
#     if i > 2:
#         if not start:
#             start = count
#     if i < 0.5:
#         if start:
#             mid_count += 1
#             if mid_count > 100:
#                 if not end:
#                     end = count
#                     time = timestamp[start - 500:end + 500]
#                     car = car1[start - 500:end + 500]
#                     start = None
#                     end = None
#
#                     plt.title('EV Power vs Timestamp - Austine Data id 7719 ' + 'Peak ID: ' + str(major_id))
#                     plt.xlabel('Time Stamp')
#                     plt.ylabel('Power (Kw)')
#                     plt.plot(time, car)
#                     figure = plt.gcf()  # get current figure
#                     figure.set_size_inches(12, 6)
#                     # when saving, specify the DPI
#                     plt.savefig("7719_data_id_austine/{}.png".format(major_id), dpi=100)
#                     plt.close(figure)
#
#                     car = None
#                     time = None
#                     major_id += 1
#     count += 1
