from Peacan_Module import DataReader
from Peacan_Module import DeepLearning

# =============== To Extract the one second Data based on IDs ====================================

# pecan = DataReader()
# pecan.dataExtractorOneSecond(
#     r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\file2.csv",
#     r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\1s_Austine_data")


# =============== Convert Extracted data's to single csv with preprocessed ====================================

# pecan.combineDataIDs(r"D:\EV_D\NY\1s_NY_data\tamil",
#                     r"D:\EV_D\NY\1s_NY_data\Combined_CSV.csv")


# =============== Run LSTM model ================================================

models = ['LSTM1', 'LSTM2', 'LSTM3', 'GRU']
dl = DeepLearning()
dl.runModel(
    r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\tain_data\Combined_CSV - Copy.csv",
    n_input=25,
    batchSize=60,
    epochs=1,
    modelName='GRU', dataset='Austine'
)

# ==================== Run Predction =========================

# dl = DeepLearning()
# dl.predictFromTheSavedModel(
#     json_path=r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\Pecan_street_Model_Iteration\Data_iteration\05_07_2021__17_00\model\model.json",
#     weights_path=r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\Pecan_street_Model_Iteration\Data_iteration\05_07_2021__17_00\model\model.h5",
#     data =r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\tain_data\Combined_CSV - Copy (2).csv"
# )
#

