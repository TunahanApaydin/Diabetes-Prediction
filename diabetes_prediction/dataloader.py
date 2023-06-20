import torch
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

class DiabetesDataset(Dataset):

    def __init__(self, csv_file, shuffle):
        """
        Arguments:
            cvs_file_path(path): 'csv' file path. \n
            shuffle(boolen): Dataset shuffling True or False. \n
        """

        self.dataset = pd.read_csv(csv_file)
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self):

        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
        
        transformer = make_column_transformer((OneHotEncoder(), ['gender', 'smoking_history']), remainder='passthrough') # one hot encoding for catagorical inputs
        transformed = transformer.fit_transform(self.dataset)
        transformed_df = pd.DataFrame(transformed, columns = transformer.get_feature_names_out())

        inputs = transformed_df.iloc[:,0:-1].values
        labels = transformed_df.iloc[:,-1].values

        input_tensor = torch.FloatTensor(inputs)
        labels_tensor = torch.FloatTensor(labels)

        return input_tensor, labels_tensor

