import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
import mlflow
import mlflow.sklearn
import torch
from matplotlib import pyplot as plt
#from torchmetrics.classification import BinaryCalibrationError
from CNN import test_sex_resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def generate_file_pairs(base_path="data/splitted_csv",
                        test_sets=3, variations=['0.00', '0.25', '0.50', '0.75', '1.00'], train_val_pairs=5):
    # Dictionary to hold all train-validation-test triples for all test sets
    all_triples = {}
    base_path = "data/cnn/cnn_splitted_data_once_augmented/"

    # Generate file names
    for test_set in range(test_sets):
        triples = []
        # Generating test set filename
        test_filename = f'{base_path}m_f_ca_nc_test_{test_set}.csv'
        for variation in variations:
            for pair in range(train_val_pairs):
                train_filename = f'{base_path}m_f_ca_nc_train_{test_set}_{variation}_{pair}.csv'
                val_filename = f'{base_path}m_f_ca_nc_val_{test_set}_{variation}_{pair}.csv'
                triples.append((train_filename, val_filename, test_filename))
        all_triples[f'test_set_{test_set}'] = triples
    return all_triples

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    #df['is_cancerous'] = df['diagnostic'].apply(lambda x: any(cancer in x for cancer in ['SCC',  'BCC', 'MEL'])).astype(int)
    X = df[["img_id"]]  # Features
    y = df[["is_cancerous"]]  # Labels
    return X, y

def load_dataset_test(filepath):
    df = pd.read_csv(filepath)

    X = df[["img_id", "gender"]]  # Features
    y = df[["is_cancerous", "gender"]]  # Labels

    X_male = X[X["gender"] == "MALE"]
    X_female = X[X["gender"] == "FEMALE"]

    
    y_male = y[y["gender"] == "MALE"]
    y_female = y[y["gender"] == "FEMALE"]

    #Make the final df

    y_male = y_male['is_cancerous']
    y_female = y_female['is_cancerous']
    y = df['is_cancerous']

    X_male = X_male[["img_id"]]
    X_female = X_female[["img_id"]]
    X = df[["img_id"]]



    # y_1 = df['diagnostic']  # Labels
    return X, y, X_female, X_male, y_female, y_male

def train_and_evaluate_model(file_triples):
    results = []
    count = 0
    img_path = "data/images/lesion_images"
    for test_set, triples in file_triples.items():
        for train_file, val_file, test_file in triples:
            X_val, y_val = load_dataset(val_file)
            X_test, y_test, X_test_female, X_test_male, y_test_female, y_test_male = load_dataset_test(test_file)
            count += 1
            with mlflow.start_run(run_name=f"cnn_{count}"):
                print('before cnn')
                y_pred, probabilities, y_pred_female, probabilities_female, y_pred_male, probabilities_male, cnn = test_sex_resnet50(train_file, val_file, test_file, img_path)

                #cnn = cnn.to(device)
                print('after cnn')
                # y_pred = y_pred.to(device)
                # y_test = y_test.to(device)  # Move labels to GPU
                print(len(y_test))
                print(len(y_pred))

                #y_score = cnn.predict_proba(X_test)[:, 1]
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_female = accuracy_score(y_test_female, y_pred_female)
                accuracy_male = accuracy_score(y_test_male, y_pred_male)
                auroc = roc_auc_score(y_test, probabilities)
                auroc_female = roc_auc_score(y_test_female, probabilities_female)
                auroc_male = roc_auc_score(y_test_male, probabilities_male)

                metrics = [
                    ("ROC", auroc),
                    ("ROC_female", auroc_female ),
                    ("ROC_male", auroc_male),
                    ("Accuracy", accuracy),
                    ("Accuracy_female", accuracy_female),
                    ("Accuracy_male", accuracy_male),
                    ("Recall", recall_score(y_test, y_pred)),
                    ("Precision", precision_score(y_test, y_pred)),
                    ("F1-score", f1_score(y_test, y_pred))
                    # ("Binary Calibration Error", BinaryCalibrationError(y_test, y_pred)) 
                ]
                for name, value in metrics:
                    mlflow.log_metric(name, value)

                variation = train_file.split('_')[-2]
                results.append({
                    "variation": variation,
                    "accuracy": accuracy,
                    "accuracy_female": accuracy_female,
                    "accuracy_male": accuracy_male,
                    "ROC": auroc,
                    "ROC_female": auroc_female,
                    "ROC_male": auroc_male,

                })
                mlflow.pytorch.log_model(cnn, "cnn") # mlflow.pytorch.log_model(cnn, "cnn")

    results_df = pd.DataFrame(results)
    results_df.to_csv('model_training_results_cnn_10_11may_R.csv', index=False)
    return results_df

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("CNN_Tester_11/05_test_with_aug")
    base_path = "data/cnn/cnn_splitted_data_once_augmented/"
    all_file_triples = generate_file_pairs(base_path=base_path)
    results = train_and_evaluate_model(all_file_triples)
