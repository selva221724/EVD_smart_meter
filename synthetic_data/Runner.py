from synthetic import SyntheticDataCreator
import pandas as pd
import matplotlib.pyplot as plt

SDC = SyntheticDataCreator()

# data = SDC.dataSynthesis(
#     pecandata=r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\1s_Austine_data\DataID_4767_1_second.csv",
#     p_timeFrame={"start": "2018-06-23 00:12:00", "end": "2018-06-24 00:12:00"},
#     SMData=r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CONVERTED DATA\LAST WEEK\conv_mci_1st_17_19.csv",
#     s_timeFrame={"start": "2021-04-18 00:00:00", "end": "2021-04-19 00:00:00"})
#
# self = SDC
#
# data.to_csv('Synthetic_1.csv')


SDC.datacombiner(path = r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\synthetic_data\Synthetic_data",
                 out_path= r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\synthetic_data\Synthetic_data\SyntheticALL.csv")