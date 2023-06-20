import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics import confusion_matrix
from dataloader import DiabetesDataset
from model_config_loader import ModelConfigs
from model_net import DiabetesPredictionModel

def inference_model(model, test_data):
    with torch.no_grad():
        model.eval()
        test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
        predictions = model(test_data_tensor)
        sm = torch.nn.Softmax(dim=0)
        probabilities = sm(predictions) 

        return predictions, probabilities

if __name__ == "__main__":
    save_confusion_matrix = True
    save_results = False
    model_config = ModelConfigs()
    model = DiabetesPredictionModel()

    model_path = model_config.saved_model_path
    best_model_path = model_config.saved_best_model_path
    inference_results_path = model_config.inference_results_path
    inference_result_file_name = model_config.inference_result_file_name

    test_dataset = DiabetesDataset(model_config.test_dataset_path, shuffle = False)
    
    test_data_input, test_data_labels = test_dataset.__getitem__()

    inputs = []
    gts = []
    outputs = []
    probs = []
    total_true = 0
    for i in range(test_dataset.__len__()):
        test_data = np.array(test_data_input[i])
        #model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path))
        predictions, probabilities = inference_model(model, test_data)
        top_p, top_class = probabilities.topk(1, dim = 0)

        if top_class.numpy()[0] == int(np.array(test_data_labels[i])):
            total_true += 1

        if not save_results:
            print("Input: {} - Gt: {} - Output: {} - Prob: {}".format(test_data, np.array(test_data_labels[i]), top_class.numpy(), top_p.detach().numpy()*100))
        
        inputs.append(test_data)
        gts.append(np.array(test_data_labels[i]))                   
        outputs.append(top_class.item())
        probs.append(top_p.item()*100)

    acc = (total_true/len(test_data_input))*100
    print(acc)

    if save_confusion_matrix:
        classes, counts = np.unique(test_data_labels, return_counts=True)

        cf_matrix = confusion_matrix(gts, outputs)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                             index = [i for i in classes],
                             columns = [i for i in classes])
        
        plt.figure(figsize = (12,7))
        sb.heatmap(df_cm, annot=True)
        plt.savefig('confusion matrix.png')

    if save_results:            
        dataset_dict = {"input": inputs,
                        "gts": gts,
                        "outputs": outputs,
                        "probs": probs}

        df_dataset_dict = pd.DataFrame(dataset_dict)


        df = pd.DataFrame(df_dataset_dict)
        df.to_csv(inference_results_path + inference_result_file_name, index=True)  