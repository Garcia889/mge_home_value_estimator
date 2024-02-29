import pandas as pd
import numpy as np
import joblib

#Lectura de datos limpios
path = "./data/"
df = pd.read_csv(path+"clean.csv")

train = df[df["group"] == "train"].drop("group", axis = 1)
test = df[df["group"] == "test"].drop(["group","SalePrice"], axis = 1)
X = train.drop(["SalePrice"], axis=1)
y = np.log(train["SalePrice"])
test_subset = test.head(10)

# Load the model from the .joblib file
loaded_model = joblib.load('src/XGB_home_model.joblib')

# Use the loaded model to make predictions on new data
predictions = loaded_model.predict(test)

import pandas as pd

# Suponiendo que 'df' es tu numpy array
df = pd.DataFrame(np.exp(predictions))
df.to_csv('predictions.csv', index=False)

test.to_csv("test_sub.csv", index=False)

