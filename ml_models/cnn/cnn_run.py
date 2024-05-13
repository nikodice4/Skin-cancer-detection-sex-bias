# Imports
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import torch
from CNN import test_sex_resnet50
import os

# Setting the device the model will run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_file_pairs(base_path="data/cnn/cnn_splitted_data_once_augmented",
                        test_sets=1, variations=["0.00", "0.25", "0.50", "0.75", "1.00"], train_val_pairs=5):
    # Here we create a dictionary that contains all 3 files. We call them triples
    all_triples = {}
    base_path = "data/cnn/cnn_splitted_data_once_augmented"

    # We name all of the files for the CNN using this pattern
    for test_set in range(test_sets):
        triples = []

        # Generating test set filename
        test_filename = f"{base_path}/m_f_ca_nc_test_{test_set}.csv"

        for variation in variations:
            for pair in range(train_val_pairs):
                train_filename = f"{base_path}/m_f_ca_nc_train_{test_set}_{variation}_{pair}.csv"
                val_filename = f"{base_path}/m_f_ca_nc_val_{test_set}_{variation}_{pair}.csv"
                triples.append((train_filename, val_filename, test_filename))
        all_triples[f"test_set_{test_set}"] = triples
    return all_triples

def load_dataset(filepath):
    # Here we load in the dataset, and just use the img_id as the feature and is_cancerous as the gold label
    df = pd.read_csv(filepath)
    X = df[["img_id"]]
    y = df[["is_cancerous"]]
    return X, y

def load_dataset_test(filepath):
    # Here we load in the test dataset. We add the sexes, as we want to test them separately 
    df = pd.read_csv(filepath)
    X = df[["img_id", "gender"]]
    y = df[["is_cancerous", "gender"]]

    X_male = X[X["gender"] == "MALE"]
    X_female = X[X["gender"] == "FEMALE"]
    
    y_male = y[y["gender"] == "MALE"]
    y_female = y[y["gender"] == "FEMALE"]

    # Make the final df with our features and gold labels
    y_male = y_male["is_cancerous"]
    y_female = y_female["is_cancerous"]
    y = df["is_cancerous"]

    X_male = X_male[["img_id"]]
    X_female = X_female[["img_id"]]
    X = df[["img_id"]]

    return X, y, X_female, X_male, y_female, y_male

def train_and_evaluate_model(file_triples):
    # Here we train and evaluate the model

          
    results = []
    count = 0
    img_path = "data/images/lesion_images"
    for test_set, triples in file_triples.items():
        for train_file, val_file, test_file in triples:
            X_val, y_val = load_dataset(val_file)
            X_test, y_test, X_test_female, X_test_male, y_test_female, y_test_male = load_dataset_test(test_file)
            count += 1

            # We start the MLFlow run
            with mlflow.start_run(run_name=f"cnn_{count}"):
                # We run the CNN and make some print statements to be sure it runs
                print("before cnn")
                y_pred, probabilities, y_pred_female, probabilities_female, y_pred_male, probabilities_male, cnn = test_sex_resnet50(train_file, val_file, test_file, img_path)
                print("after cnn")

                accuracy = accuracy_score(y_test, y_pred)
                accuracy_female = accuracy_score(y_test_female, y_pred_female)
                accuracy_male = accuracy_score(y_test_male, y_pred_male)
                auroc = roc_auc_score(y_test, probabilities)
                auroc_female = roc_auc_score(y_test_female, probabilities_female)
                auroc_male = roc_auc_score(y_test_male, probabilities_male)

                # We log the metrics to MLFlow
                metrics = [
                    ("AUROC", auroc),
                    ("AUROC_female", auroc_female),
                    ("AUROC_male", auroc_male),
                    ("Accuracy", accuracy),
                    ("Accuracy_female", accuracy_female),
                    ("Accuracy_male", accuracy_male),
                    ("Recall", recall_score(y_test, y_pred)),
                    ("Precision", precision_score(y_test, y_pred)),
                    ("F1-score", f1_score(y_test, y_pred))
                ]

                for name, value in metrics:
                    mlflow.log_metric(name, value)

                # We define what we want in the output file
                # Variation shows the sex ratio of the training data
                variation = train_file.split("_")[-2]
                results.append({
                    "variation": variation,
                    "accuracy": accuracy,
                    "accuracy_female": accuracy_female,
                    "accuracy_male": accuracy_male,
                    "auroc": auroc,
                    "auroc_female": auroc_female,
                    "auroc_male": auroc_male,
                    "Recall": recall_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred),
                    "F1-score": f1_score(y_test, y_pred),
                })
                
                # Here we log the model to MLFlow (if we want to)
                mlflow.pytorch.log_model(cnn, "cnn")

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/results/cnn_results/quick_test.csv", index=False)
    return results_df

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5050")
    mlflow.set_experiment("CNN_Augmented")
    base_path = "data/cnn/cnn_splitted_data_once_augmented"
    all_file_triples = generate_file_pairs(base_path=base_path)
    results = train_and_evaluate_model(all_file_triples)