from Peacan_Module import DataReader
from Peacan_Module import DeepLearning
# =============== To Extract the one second Data based on IDs ====================================

# pecan = DataReader()
# pecan.dataExtractorOneSecond(
#     r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\file2.csv",
#     r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\1s_Austine_data")


# =============== Convert Extracted data's to single csv with preprocessed ====================================

# pecan.combineDataIDs(r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\1s_Austine_data",
#                     r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\tain_data\Combined_CSV.csv")
#



# =============== Run LSTM model ================================================

dl = DeepLearning()
dl.runModel(r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\out - Copy.csv",
            n_input=25,
            batchSize=60,
            epochs=3,
            modelName='LSTM'
            )

