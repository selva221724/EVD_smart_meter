from synthetic import SyntheticDataCreator
import pandas as pd

SDC = SyntheticDataCreator()

data = SDC.dataSynthesis(
    pecandata=r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\synthetic_data\data\DataID_661_1_second.csv",
    p_timeFrame={"start": "2018-05-12 00:00:00", "end": "2018-05-13 00:00:00"},
    SMData=r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\synthetic_data\data\floor1.csv",
    s_timeFrame={"start": "2021-03-17 00:00:00", "end": "2021-03-18 00:00:00"})

self = SDC

data.to_csv('Synthetic_1.csv')