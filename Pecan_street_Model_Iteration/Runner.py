from Peacan_Module import DataReader
from Peacan_Module import DeepLearning
import time

# =============== To Extract the one second Data based on IDs ====================================

# pecan = DataReader()
# pecan.dataExtractorOneSecond(
#     r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\file2.csv",
#     r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\1s_Austine_data")


# =============== Convert Extracted data's to single csv with preprocessed ====================================

# pecan.combineDataIDs(r"D:\EV_D\NY\1s_NY_data\tamil",
#                     r"D:\EV_D\NY\1s_NY_data\Combined_CSV.csv")


# =============== Run LSTM model ================================================
#
models = ['LSTM1', 'LSTM2', 'LSTM3', 'GRU']
dl = DeepLearning()
dl.runModel(


    r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\tain_data\Combined_CSV - Copy.csv",
    n_input=60,
    batchSize=60,
    epochs=5,
    modelName='LSTM2', dataset='Austine',
    # loadFromCheckPoint = r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\Pecan_street_Model_Iteration\Data_iteration\12_07_2021__22_35\checkpoints\05.hdf5"

)

# ============== Run DL model as a mutiple Iteration ====================================
# import itertools
#
# n_input = [25, 60]
# batch_size = [60]
# epochs = [1, 10, 20]
# get_combination = list(itertools.product(*[n_input, batch_size, epochs]))
#
# for i in get_combination:
#     print("=====================================================================")
#     print("n_inputs: "+str(i[0])+" batch size: "+str(i[1])+" epochs: "+str(i[2]))
#     print("=====================================================================")
#     dl = DeepLearning()
#     dl.runModel(
#         r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\tain_data\Combined_CSV - Copy.csv",
#         n_input=i[0],
#         batchSize=i[1],
#         epochs=i[2],
#         modelName='LSTM2', dataset='Austine'
#     )
#     time.sleep(60)

# ==================== Run Predction =========================

# dl = DeepLearning()
# dl.predictFromTheSavedModel(
#     json_path=r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\Pecan_street_Model_Iteration\Data_iteration\05_07_2021__17_00\model\model.json",
#     weights_path=r"C:\Users\TamilS\Documents\Python Scripts\EV\EVD_smart_meter\Pecan_street_Model_Iteration\Data_iteration\05_07_2021__17_00\model\model.h5",
#     data =r"C:\Users\TamilS\Documents\Python Scripts\EV\EV DETECTION\CNN\Pecan_street_data_set\DATAPORT\Austin\tain_data\Combined_CSV - Copy (2).csv"
# )
#
