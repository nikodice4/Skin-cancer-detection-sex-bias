import mlflow
import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import PredefinedSplit
import warnings

def generate_file_pairs(base_path="data/lr/lr_splitted_aug_data",
                        test_sets=5, variations=["0.00", "0.25", "0.50", "0.75", "1.00"], train_val_pairs=5):
    # Here we create a dictionary that contains all 3 files. We call them triples
    all_triples = {}

    # We name all of the files for the LR using this pattern
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


def load_dataset_tv(filepath):
    # Here we load the dataset for training and validation 
    df = pd.read_csv(filepath)
    X = df[["F2", "F11", "sat_var", "blue_veil_pixels", "avg_green_channel", 
            "mean_asymmetry", "F1", "average_hue", "compactness_x", "dom_hue"]]
    y = df["is_cancerous"] 

    return X, y


def load_dataset_test(filepath):
    # Loading the dataset for test. We test for females and males separately
    df = pd.read_csv(filepath)

    X = df[["F2", "F11", "sat_var", "blue_veil_pixels", "avg_green_channel", 
            "mean_asymmetry", "F1", "average_hue", "compactness_x", "dom_hue", "gender"]]
    y = df[["is_cancerous", "gender"]]

    X_male = X[X["gender"] == "MALE"]
    X_female = X[X["gender"] == "FEMALE"]
    
    y_male = y[y["gender"] == "MALE"]
    y_female = y[y["gender"] == "FEMALE"]

    # Make the final df with our features and gold labels
    y_male = y_male["is_cancerous"]
    y_female = y_female["is_cancerous"]
    y = df["is_cancerous"]

    X_male = X_male[["F2", "F11", "sat_var", "blue_veil_pixels", "avg_green_channel", 
                     "mean_asymmetry", "F1", "average_hue", "compactness_x", "dom_hue"]]
    X_female = X_female[["F2", "F11", "sat_var", "blue_veil_pixels", "avg_green_channel", 
                         "mean_asymmetry", "F1", "average_hue", "compactness_x", "dom_hue"]]
    X = df[["F2", "F11", "sat_var", "blue_veil_pixels", "avg_green_channel", 
            "mean_asymmetry", "F1", "average_hue", "compactness_x", "dom_hue"]]

    return X, y, X_female, X_male, y_female, y_male



def train_and_evaluate_model(file_triples):
    # Here we train and evaluate the model
    results = []
    count = 0 
    for test_set, triples in file_triples.items():
        for train_file, val_file, test_file in triples:
            count += 1
                    
            X_train, y_train = load_dataset_tv(train_file)
            X_val, y_val = load_dataset_tv(val_file)
            X_test, y_test, X_test_female, X_test_male, y_test_female, y_test_male = load_dataset_test(test_file)

            # This is where we start the MLFlow run
            with mlflow.start_run(run_name=f"logistic_regression_{count}"):
                # We define the model, standardise the data and create a pipeline for it
                logistic = LogisticRegression()
                scaler = StandardScaler()
                pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

                # We combine the training and validation data to create the predefined split
                # using our own folds for the cross-validation in GridSearchCV
                combined_X = np.vstack((X_train, X_val))
                combined_y = pd.concat([y_train, y_val])
                split_index = [-1] * len(X_train) + [0] * len(X_val)
                pds = PredefinedSplit(test_fold=split_index)

                # We define the parameters for the GridSearchCV
                # We create two dictionaries for the parameter combiantions, as solver "lbfgs"
                # can ONLY use l2 penalty, and solver "liblinear" can use both l1 and l2
                parameters = [
                    {"logistic__solver": ["liblinear"],
                    "logistic__penalty": ["l1", "l2"],
                    "logistic__fit_intercept": [True, False],
                    "logistic__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5],
                    "logistic__class_weight": ["balanced", None],
                    "logistic__max_iter": [50, 100, 500, 1000]},
                    {"logistic__solver": ["lbfgs"],
                    "logistic__penalty": ["l2", None],
                    "logistic__fit_intercept": [True, False],
                    "logistic__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5],
                    "logistic__class_weight": ["balanced", None],
                    "logistic__max_iter": [50, 100, 500, 1000]}
                    ]

                # We perform the GridSearchCV and define it as "search)"
                grid_search = GridSearchCV(estimator=pipe,
                                           param_grid=parameters,
                                           cv=pds,
                                           scoring="accuracy")
                search = grid_search.fit(combined_X, combined_y)

                # We predict the whole test data, and for male and female separately
                # This is to find the accuracy, and other metrics
                y_pred = search.predict(X_test)
                y_pred_male = search.predict(X_test_male)
                y_pred_female = search.predict(X_test_female)

                # We predict the probabilities to get the AUROC score
                y_score = search.predict_proba(X_test)[:, 1]
                y_score_female = search.predict_proba(X_test_female)[:, 1]
                y_score_male = search.predict_proba(X_test_male)[:, 1]
                auroc = roc_auc_score(y_test, y_score)
                auroc_female = roc_auc_score(y_test_female, y_score_female)
                auroc_male = roc_auc_score(y_test_male, y_score_male)

                # We calculate the accuracy for all groups
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_female = accuracy_score(y_test_female, y_pred_female)
                accuracy_male = accuracy_score(y_test_male, y_pred_male)

                # We print the results
                print("Accuracy for all groups:", accuracy)
                print("Best gridsearch score", search.best_score_)
                print(f"Penalty {search.best_params_['logistic__penalty']}, \
                    Solver {search.best_params_['logistic__solver']}, \
                    Fit intercept {search.best_params_['logistic__fit_intercept']}, \
                    C {search.best_params_['logistic__C']}, \
                    Class weight {search.best_params_['logistic__class_weight']}, \
                    Max iterations {search.best_params_['logistic__max_iter']}")

                    
                # Code to run confusion matrices for the whole test data 
                # cm = confusion_matrix(y_test, y_pred)
                # cm = ConfusionMatrixDisplay(cm)
                # cm = cm.plot()
                # plt.savefig(f"../../analysis/plots/cm_plots/confusion_matrix_{count}.png")
                # plt.show()

                # We log the confusion matrices to MLFlow
                # mlflow.log_artifacts(f"confusion_matrix_{count}.png", "confusion_matrices")


                # We log the metrics and parameters to MLFlow
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
                
                # We log parameters
                mlflow.log_params(search.best_params_)

                # We define what we want in the output file
                # Variation shows the sex ratio of the training data
                variation = train_file.split("_")[-2] 
                results.append({
                    "variation": variation,
                    "accuracy": accuracy,
                    "accuracy_female": accuracy_female,
                    "accuracy_male": accuracy_male,
                    "auroc": roc_auc_score(y_test, y_score), 
                    "auroc_female": roc_auc_score(y_test_female, y_score_female),
                    "auroc_male": roc_auc_score(y_test_male, y_score_male),
                    "Recall": recall_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred),
                    "F1-score": f1_score(y_test, y_pred),
                    "best_params": search.best_params_,
                    "best_score": search.best_score_
                })


                # Here we log the model to MLFlow (if we want to)
                # mlflow.sklearn.log_model(search, "Logistic_regression_model")

    # We convert the results to a dataframe
    results_df = pd.DataFrame(results)
    # We save the results to a csv file
    results_df.to_csv("data/results/lr_results/final_lr_results.csv", index=False)
    return results_df


if __name__ == "__main__":
    # We define where we can find the logged models, parameters, metrics etc., and under a specific name
    mlflow.set_tracking_uri("http://127.0.0.1:5048")
    mlflow.set_experiment("Logistic_Regression_Augmented")

    # For code reproducibility, we set a seed"
    np.random.seed(4)

    base_path = "data/lr/lr_splitted_aug_data"
    all_file_triples = generate_file_pairs(base_path)
    results = train_and_evaluate_model(all_file_triples)