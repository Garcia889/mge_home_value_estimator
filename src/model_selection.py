#Prueba de modelos
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor
from data_preprocessor import dataPreprocessor
import logging

#Variables
homes_input = "../data/train.csv"
homes_output = "../data/clean_train.csv"
logs_output = "../logs/training_logs.csv"

#Train data preprocessing
procesador = dataPreprocessor(homes_input, homes_output, logs_output)
procesador.procesar_csv_homes()

#Logging configuration
logging.basicConfig(
            filename=logs_output,
            level=logging.INFO,
            filemode='a',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#Reading training data
train = pd.read_csv(homes_output)
X = train.drop(["SalePrice"], axis=1)
y = np.log(train["SalePrice"])

def basemodel(X, y, cv=5, scoring=["r2", "neg_mean_squared_error"]):
    """
    Realiza la selección de modelos y evalúa su desempeño utilizando validación cruzada.

    Parámetros:
    - X: array-like, shape (n_samples, n_features)
        Conjunto de datos de entrada.
    - y: array-like, shape (n_samples,)
        Etiquetas o valores objetivo.
    - cv: int, opcional (default=5)
        Número de folds en la validación cruzada.
    - scoring: list, opcional (default=["r2", "neg_mean_squared_error"])
        Métricas de evaluación utilizadas para evaluar el desempeño de los modelos.

    Retorna:
    None
    """
    regressions = [
        ("Linear", LinearRegression()),
        ("KNN", KNeighborsRegressor(n_neighbors=5)),
        ("XGBRegressor", XGBRegressor())
    ]
    
    logging.info("Model scores")
    for name, model in regressions:
        base_model = cross_validate(model, X, y, cv=cv, scoring=scoring, error_score='raise')
        scores_r2 = np.mean(base_model['test_r2'])
        mse_scores = -np.mean(base_model['test_neg_mean_squared_error'])
        print(f"{name} - R2 Scores: {scores_r2}, MSE Scores: {mse_scores}")
        logging.info(f"{name} - R2 Scores: {scores_r2}, MSE Scores: {mse_scores}")

basemodel(X, y, cv=5, scoring=["r2", "neg_mean_squared_error"])

