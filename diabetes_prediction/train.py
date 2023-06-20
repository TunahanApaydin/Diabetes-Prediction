import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import DiabetesDataset
from model_config_loader import ModelConfigs
from model_net import DiabetesPredictionModel

class TrainDiabetesModel:
    def __init__(self):
        self.model_config = ModelConfigs()
        self.model = DiabetesPredictionModel()

        self.class_weights = []
        self.training_loss = []
        self.validation_loss = []
        self.training_acc = []
        self.validation_acc = []

        self.train_data_input = torch.Tensor
        self.train_data_labels = torch.Tensor
        self.valid_data_input = torch.Tensor
        self.valid_data_labels = torch.Tensor
        
    def train(self):
        print(self.model)

        loss_fn = nn.CrossEntropyLoss(weight = self.class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr = self.model_config.learning_rate)

        train_loader = DataLoader(list(zip(self.train_data_input, self.train_data_labels)),
                                  shuffle = True, batch_size = self.model_config.batch_size)
        validation_loader = DataLoader(list(zip(self.valid_data_input, self.valid_data_labels)),
                                       shuffle = False, batch_size = self.model_config.batch_size)
        
        min_valid_loss = np.inf
        for epoch in range(self.model_config.number_of_epoch):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for Xbatch, ybatch in train_loader:
                if torch.cuda.is_available():
                    Xbatch, ybatch = Xbatch.cuda(), ybatch.cuda()
                
                optimizer.zero_grad()

                y_pred = self.model(Xbatch)
                ybatch = [output for output in ybatch]
                ybatch = torch.tensor(ybatch)
                
                loss = loss_fn(y_pred, ybatch.long())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                _, pred = torch.max(y_pred.data, 1)
                train_total += ybatch.size(0)
                train_correct += pred.eq(ybatch).sum().item()

            self.training_loss.append(train_loss/len(train_loader))
            self.training_acc.append(100.*train_correct/train_total)

            print("Epoch: {} - Training Loss: {}".format(epoch+1, train_loss/len(train_loader)))
            print("Epoch: {} - Training Acc: {}".format(epoch+1, 100.*train_correct/train_total))

            valid_loss = 0.0
            valid_correct = 0
            valid_total = 0
            self.model.eval()
            for Xbatch, ybatch in validation_loader:     

                y_pred = self.model(Xbatch)

                ybatch = [output for output in ybatch]
                ybatch = torch.tensor(ybatch)
                loss = loss_fn(y_pred, ybatch.long())

                valid_loss += loss.item()

                _, pred = torch.max(y_pred.data, 1)
                valid_total += ybatch.size(0)
                valid_correct += pred.eq(ybatch).sum().item()

            self.validation_loss.append(valid_loss/len(validation_loader))
            self.validation_acc.append(100.*valid_correct/valid_total)

            print("Epoch: {} - Validation Loss: {}".format(epoch+1, valid_loss/len(validation_loader)))
            print("Epoch: {} - Validation Acc: {}".format(epoch+1, 100.*valid_correct/valid_total))

            if min_valid_loss > valid_loss:
                print("Saving best model at epoch: {}".format(epoch+1))
                min_valid_loss = valid_loss
                self.save_checkpoint(is_best = True)

    def save_checkpoint(self, is_best):
        if not is_best:
            #torch.save(self.model, self.model_config.save_path + self.model_config.model_name + ".pt")
            torch.save(self.model.state_dict(), self.model_config.save_path + self.model_config.model_name + ".pt")
        else:
            torch.save(self.model.state_dict(), self.model_config.save_path + self.model_config.best_model_name + ".pt")
    
    def save_loss_graph(self, visualize):
        loss_fig = plt.figure(figsize=(50,50))
        loss_fig.plot(self.training_loss, label='train_loss',)
        loss_fig.plot(self.validation_loss, label='validation_loss')
        loss_fig.xlabel("Number of the Epoch")
        loss_fig.ylabel("Loss")
        loss_fig.legend()
        loss_fig.savefig("graphs/loss_graphs/loss_graph_"+ \
                    str(self.model_config.number_of_epoch) + "e_" + \
                    str(self.model_config.learning_rate) + "lr" + ".png")
        if visualize:
            loss_fig.show()
    
    def save_acc_graph(self, visualize):
        plt.plot(self.training_acc, label='train_acc')
        plt.plot(self.validation_acc, label='validation_acc')
        plt.xlabel("Number of the Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("graphs/acc_graphs/acc_graph_"+ \
                    str(self.model_config.number_of_epoch) + "e_" + \
                    str(self.model_config.learning_rate) + "lr" + ".png")
        if visualize:
            plt.show()
    
    def save_loss_acc_graphs(self, visualize):
        
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(self.training_loss, label='train loss')
        ax[0].plot(self.validation_loss, label='validation loss')
        ax[0].set_xlabel("Number of the Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
   
        ax[1].plot(self.training_acc, label='train acc')
        ax[1].plot(self.validation_acc, label='validation acc')
        ax[1].set_xlabel("Number of the Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        

        ax[0].set_title('Training and Validation Loss')
        ax[1].set_title('Training and Validation Accuracy')
        
        fig.suptitle('Missing Teeth Model Analysis Graphs')

        fig.tight_layout()

        plt.savefig("graphs/loss_acc_graphs/"+ \
                    str(self.model_config.number_of_epoch) + "e_" + \
                    str(self.model_config.learning_rate) + "lr" + ".png", dpi=200)
        
        if visualize:
            plt.show()

if __name__ == "__main__":
    save_class_counts = False
    training = TrainDiabetesModel()

    train_dataset = DiabetesDataset(training.model_config.train_dataset_path, shuffle = False)
    
    validation_dataset = DiabetesDataset(training.model_config.validation_dataset_path, shuffle = False)

    training.train_data_input, training.train_data_labels = train_dataset.__getitem__()
    training.valid_data_input, training.valid_data_labels = validation_dataset.__getitem__()

    classes, counts = np.unique(training.train_data_labels, return_counts=True)
    for index in range(len(classes)):
        training.class_weights.append(round(1 - (counts[index]/counts.sum()), 6))
    training.class_weights = torch.Tensor(training.class_weights)

    if save_class_counts:
        plt.bar(classes, counts)
        plt.xlabel("Classes")
        plt.ylabel("Counts")
        plt.savefig("class_counts")
    
    training.train()
    training.save_checkpoint(is_best = False)
    training.save_loss_acc_graphs(visualize = False)
    # training.save_loss_graph(visualize = False)
    # training.save_acc_graph(visualize = False)