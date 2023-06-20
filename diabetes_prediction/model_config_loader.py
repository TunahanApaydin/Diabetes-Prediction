import configparser

class ModelConfigs:
    def __init__(self):
        config  = configparser.ConfigParser()     
        config.readfp(open(r'diabetes_prediction_config.ini'))

        self.learning_rate = config.getfloat("HYPER PARAMETERS", "learning_rate")
        self.number_of_epoch = config.getint("HYPER PARAMETERS", "number_of_epoch")
        self.batch_size = config.getint("HYPER PARAMETERS", "batch_size")

        self.train_dataset_path = config.get("DATASET PARAMETERS", "train_dataset_path")
        self.validation_dataset_path = config.get("DATASET PARAMETERS", "validation_dataset_path")

        self.save_path = config.get("SAVE PARAMETERS", "save_path")
        self.model_name = config.get("SAVE PARAMETERS", "model_name")
        self.best_model_name = config.get("SAVE PARAMETERS", "best_model_name")

        self.saved_model_path = config.get("INFERENCE PARAMETERS", "saved_model_path")
        self.saved_best_model_path = config.get("INFERENCE PARAMETERS", "saved_best_model_path")
        self.test_dataset_path = config.get("INFERENCE PARAMETERS", "test_dataset_path")
        self.inference_results_path = config.get("INFERENCE PARAMETERS", "inference_results_path")
        self.inference_result_file_name = config.get("INFERENCE PARAMETERS", "inference_result_file_name")


if __name__  == "__main__":
    model_configs = ModelConfigs()