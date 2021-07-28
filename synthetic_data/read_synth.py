import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\synthetic_data\Synthetic_data\SyntheticALL.csv",dtype=object)

power = data['total_power']
power = [float(i) for i in power]
plt.plot(range(len(power)),power)