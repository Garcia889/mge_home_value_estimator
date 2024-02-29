#Prueba de hiperparametros del mejor modelo
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

homes_output = "../data/clean_train.csv"

train = pd.read_csv(homes_output)
X = train.drop(["SalePrice"], axis=1)
y = np.log(train["SalePrice"])

#print("Prueba de hiperparametros del mejor modelo")
xgb=XGBRegressor().fit(X,y)
xgb.get_params()

xgb_new_params = {"n_estimators": [100, 500, 1000],
                  "max_depth": [3, 5, 7],
                  "learning_rate": [0.01, 0.05, 0.1],
                  "subsample": [0.5, 0.7, 0.9],
                  'colsample_bytree': [0.5, 0.7, 0.9],
                  'gamma': [0, 0.1, 0.2]
                  }

xgb_find_= GridSearchCV(xgb,xgb_new_params,verbose=True,cv=5,n_jobs=-1).fit(X,y)
best=xgb_find_.best_params_
print(best)

#XGB_final=XGBRegressor(colsample_bytree=0.7, gamma=0, learning_rate=0.05, max_depth=3, n_estimators=1000, subsample=0.9).fit(X,y)
XGB_final=XGBRegressor(best).fit(X,y)
XGB_final_best=cross_validate(XGB_final,X,y, scoring=["r2", "neg_mean_squared_error"],cv=3)
scores_r2 = np.mean(XGB_final_best['test_r2'])
mse_scores = -np.mean(XGB_final_best['test_neg_mean_squared_error'])
print(scores_r2)
print(mse_scores)

# Guarda el modelo con joblib
#joblib.dump(XGB_final, 'XGB_home_model.joblib')
