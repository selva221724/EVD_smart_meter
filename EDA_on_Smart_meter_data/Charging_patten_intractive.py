import pandas as pd
import altair as alt

data_path = r"C:\Users\sivashankar.palraj\PycharmProjects\NY\DataID_1222_1_second.csv"

sourceData = pd.read_csv(data_path)

sourceData = sourceData.set_index('localminute')

sourceData.index = pd.to_datetime(sourceData.index, format='%Y-%m-%d %H:%M:%S-%f')
sourceData = sourceData.sort_index().loc['2019-05-01 00:00:00':'2019-05-01 01:00:00']

source = pd.DataFrame({
  'Time Stamp': sourceData.index,
  'Power (Kw)': sourceData['car1']
})
alt.renderers.enable('altair_viewer')
alt.Chart(source).mark_line().encode(
    y='Power (Kw)',
    x='Time Stamp'
).interactive().show()#save('NY_1222.html')