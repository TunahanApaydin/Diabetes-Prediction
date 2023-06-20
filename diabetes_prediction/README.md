The aim of the project is to predict diabetes. 'Diabetes prediction dataset', which can be accessed from the Kaggle platform(https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset?resource=download), was used as the dataset. The dataset contains information in 8 categories. Some of these categories contain categorical data. For a better model training, one hot encoding process was applied to the categorical data. Thus, the number of inputs of the model increased to 15.

### Creating Pytorch Model
The 6-layer model with 15 inputs and 2 outputs was created using Pytorch. The model network can be viewed here:
```bash
model_net.py
```
### Creating Dataset
The dataset used to train the model consists of 10000 data. In order to better analyze the model performance, the data set was divided into train, validation and test parts. Check:
```bash
split_dataset.py
```

### Creating Dataloader
The dataset is transferred to the model with the ```dataloader.py``` code. This code reads data from csv file. It applies one hot encoding to the categorical data. Returns input and output data as tensor.

### Hyperparameter Reader
All parameters and paths used for model training and inference are read from the ```diabetes_prediction_config.ini``` config file with the code ```model_config_loader.py```.

### Model Training
Now we have the model and the dataset but for a good model, it is important that the dataset is balanced.
![class_counts](https://github.com/TunahanApaydin/Pytorch-Examples/assets/79514917/1dbf0a54-dae9-4f6f-81dc-402622cb6784)

As you can see, the data set is imbalanced. The 'class weight balancer' operation can be applied for training unbalanced data sets. The following formula can be used to achieve this:
```bash
class weight = 1 - (class counts/total count)
```
And the weights obtained for all classes are given as a parameter to the loss function. For example ```loss_fn = nn.CrossEntropyLoss(weight = class_weights)```

---
In the training code, the model is trained as much as the number of epochs set in the config file. Loss and accuracy are calculated on training and validation sets. These calculated values are graphically saved in the ```graphs/loss_acc_graphs``` folder at the end of the training. The best model during the training and the last model at the end of the training are saved in the ```saved_model``` folder.

Example loss and accuracy graph:
![20e_0 001lr](https://github.com/TunahanApaydin/Pytorch-Examples/assets/79514917/fa9d4fcb-209c-4626-80a1-fea37da84807)


### Model Inference
In the Inference code, the model is tested on the test data read through the dataloader.py code. The performance on the test data is calculated and the inference results are saved in the ```inference_results``` folder in csv format. The confusion matrix graph is saved in the ```graphs``` folder. This chart will be useful for analyzing model performance.

Example confusion matrix graph:
![confusion matrix](https://github.com/TunahanApaydin/Pytorch-Examples/assets/79514917/f82e5008-4dc9-449d-8066-e3f07007cf4d)

