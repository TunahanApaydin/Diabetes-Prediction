import pandas as pd

csv_file = "/your_path/diabetes_prediction_dataset.csv"
ds_root_save_path = "/your_path/diabetes_dataset/"

dataset = pd.read_csv(csv_file)
dataset = dataset.sample(frac=1).reset_index(drop=True)

train_dataset = dataset.iloc[0:int((len(dataset)*60)/100),:]
validation_dataset = dataset.iloc[int((len(dataset)*60)/100):int((len(dataset)*80)/100),:]
test_dataset = dataset.iloc[int((len(dataset)*80)/100):-1,:]

train_dataset.to_csv(ds_root_save_path + "train/" + "train_dataset.csv", index=False)
validation_dataset.to_csv(ds_root_save_path + "validation/" + "validation_dataset.csv", index=False) 
test_dataset.to_csv(ds_root_save_path + "test/" + "test_dataset.csv", index=False) 




