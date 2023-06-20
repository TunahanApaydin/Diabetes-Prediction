The aim of the project is to predict diabetes. 'Diabetes prediction dataset', which can be accessed from the Kaggle platform(https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset?resource=download), was used as the dataset.
The dataset contains information in 8 categories. Some of these categories contain categorical data.
For a better model training, one hot encoding process was applied to the categorical data. Thus, the number of inputs of the model increased to 15.

## Creating Pytorch Model
The 6-layer model with 15 inputs and 2 outputs was created using Pytorch. The model network can be viewed here:
```bash
model_net.py
```
## Creating Dataset
The dataset used to train the model consists of 10000 data. In order to better analyze the model performance, the data set was divided into train, validation and test parts. Check:
```bash
split_dataset.py
```

## Creating Dataloader
The dataset is transferred to the model with the ```dataloader.py``` code. This code reads data from csv file. It applies one hot encoding to the categorical data. Returns input and output data as tensor.

## Hyperparameter Reader
All parameters and paths used for model training and inference are read from the ```diabetes_prediction_config.ini``` config file with the code ```model_config_loader.py```.

## Model Training
Now we have the model and the dataset. In the training code, the model is trained as much as the number of epochs set in the config file. Loss and accuracy are calculated on training and validation sets. These calculated values are graphically saved in the ```graphs/loss_acc_graphs``` folder at the end of the training. The best model during the training and the last model at the end of the training are saved in the ```saved_model``` folder.

Sample loss and accuracy graph:

