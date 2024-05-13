import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score


    #df = pd.read_csv("data/data_ready.csv")

#data = pd.read_csv("data/extracted_features/feature_group9_all.csv")
#data2 = pd.read_csv("data/extracted_features/feature_group4_all.csv")
#metadata = pd.read_csv('data/df_pad_ufes.csv')


def logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
            pca = PCA()

            scaler = StandardScaler()

            logistic = LogisticRegression(max_iter=10000)#.fit(X_train, y_train)

            pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])


            # TODO: Find flere parametre 
            parameters = {
                    "pca__n_components": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "logistic__penalty": ["l1", "l2"],
                    "logistic__C": [0.1, 0.5, 1, 2, 5, 10]
                }


            grid_search = GridSearchCV(estimator=pipe, param_grid=parameters, cv=5, scoring='accuracy')



            search = grid_search.fit(X_train,y_train)
            mlflow.log_param("Best parameters using gridsearch", search.best_params_)
            mlflow.log_metric("Best score using gridsearch", search.best_score_)



            y_pred = search.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy for all groups:", accuracy)
            print("Best gridsearch score", search.best_score_)
            print("Number of components PCA", search.best_params_["pca__n_components"])
            print("Penalty", search.best_params_["logistic__penalty"])
            print("C", search.best_params_["logistic__C"])


            metrics = [
                ("ROC", roc_auc_score(y_test, y_pred), []),
                ("Accuracy", accuracy_score(y_test, y_pred), []),
                ("Recall", recall_score(y_test, y_pred), []),
                ("Precision", precision_score(y_test, y_pred), [])
            ]

            for name, value, _ in metrics:
                mlflow.log_metric(name, value)


            mlflow.sklearn.log_model(search, "Logistic_regression")

