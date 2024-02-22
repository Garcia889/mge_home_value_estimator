##Bibliotecas
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

##Leemos y combinamos datos de entrenamiento y prueba
path = "../data/"
train = pd.read_csv(path+"train.csv").drop("Id",axis=1)
test = pd.read_csv(path+"test.csv").drop("Id",axis=1)
sub = pd.read_csv(path+"sample_submission.csv")

def con_cat(train, test):
    """
    Concatena dos dataframes verticalmente y agrega una columna 'group'
    para indicar la fuente de cada fila.
    
    Parámetros:
    train (pandas.DataFrame): El primer dataframe a concatenar.
    test (pandas.DataFrame): El segundo dataframe a concatenar.
    """
    df1, df2 = train.copy(), test.copy()
    df1["group"] = "train"
    df2["group"] = "test"
    
    return pd.concat([df1, df2], axis=0, ignore_index=True)

df = con_cat(train, test)

##Separamos variables categóricas y numéricas
categorical_cols = [col for col in df.select_dtypes(include='object').columns
                    if df[col].nunique() < 4]

cardinal_cols = [col for col in df.select_dtypes(include='object').columns
                 if df[col].nunique() >= 4]

cat_cols = [col for col in df.select_dtypes(include='object').columns]

numerical_col = [col for col in df.select_dtypes(include=['float64', 'int64']).columns
                 if col != "SalePrice" and df[col].nunique() > 1]

num_but_car = [col for col in df.select_dtypes(include=['float', 'int']).columns
               if df[col].nunique() < 4 and col != "SalePrice"]

total_Cat = cardinal_cols + categorical_cols
total_Num = num_but_car + numerical_col
other = [col for col in df.columns if col not in total_Cat + total_Num]

#print("Categóricas:", len(total_Cat), total_Cat)
#print("Numéricas:", len(total_Num), total_Num)
#print("Otras variables:", len(other), other)


##En variables categóricas, revisamos valores nulos y 
##para este ejercicio les asignamos cero

def nan_detected(dataframe, col_names):
    nan_values = dataframe[col_names].isnull().sum()
    nan_ratio = nan_values / len(dataframe[col_names])
    #print(f"{col_names} Nan values: {nan_values}\n {col_names} Nan ratio: {nan_ratio}")


def nan_fixcat(dataframe, col_names):
    for col in col_names:
        if dataframe[col].isnull().sum() > 0:
            dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)
    print("NaN values fixed successfully.")
    return dataframe

df_detected = nan_detected(df, total_Cat)
df_fixed = nan_fixcat(df, total_Cat)

##En variables numéricas, revisamos valores nulos y si existen les asignamos la media
##Imprimimos número de correcciones de cada variable numérica
def nan_fixnum(dataframe, col_names):
    fixes = dataframe[col_names].fillna(dataframe[col_names].median(), inplace=True)
    if fixes is not None and fixes > 0:
        print(fixes)


for col in total_Num:
    nan_fixnum(df, col)

##Eliminamos variables categóricas que más del 90% de sus registros tengan el mismo valor
def dominant_values(dataframe, col):
    dominant_ratio = (dataframe[col].value_counts() / len(dataframe[col])) * 100
    #print(f"Dominant R for {col}: {dominant_ratio}")

    threshold = 90

    if dominant_ratio.max() > threshold:
        dataframe.drop(col, axis=1, inplace=True)
        return f"{col} 90% dominant value, dropped successfully."

print("Dominant columns:")
for col in total_Cat:
    has_dominant = dominant_values(df, col)
    if has_dominant is not None:
        print(has_dominant)

##Eliminamos variables numéricas que tengan una correlación menor a 0.1 
##con la variable objetivo
        
'''
correlation = df.corr(numeric_only=True)
correlation = correlation["SalePrice"].sort_values(ascending=False)
low_correlation = correlation[correlation < 0.1]
low_correlation_cols = low_correlation.index
low_correlation_cols = low_correlation.index.drop("YrSold")
df.drop(low_correlation_cols, axis=1, inplace=True)
print("Low correlation columns:")
for col in low_correlation_cols:
    print(col, "low correlation, dropped successfully.")

df.drop(low_correlation_cols, axis=1, inplace=True)
print("Low correlation columns:")
for col in low_correlation_cols:
    print(col, "low correlation, dropped successfully.")
'''


##Agregamos nuevas variables como Edad de la casa o número total de baños
def new_features(dataframe):
    dataframe['HouseAge'] = dataframe['YrSold'] - dataframe['YearBuilt']
    dataframe['OverallQualityCondition'] = dataframe['OverallQual'] + dataframe['OverallCond']
    dataframe['TotalBathrooms'] = dataframe['BsmtFullBath'] + dataframe['FullBath']
    return dataframe

df = new_features(df)

categorical_cols = [col for col in df.columns if
                    df[col].dtypes == 'object' and df[col].nunique() < 4]
cardinal_cols = [col for col in df.columns if
                 df[col].dtypes == 'object' and df[col].nunique() >= 4]
numerical_col = [col for col in df.columns if
                 (df[col].dtypes == 'float64' or df[col].dtypes == 'int64') and (col != "SalePrice") and df[
                     col].nunique() > 1 and (col != "Id")]

num_but_car = [col for col in df.columns if
               (df[col].nunique() < 4 and (df[col].dtypes == 'float' or df[col].dtypes == 'int')) and (
                       col != "SalePrice")]
total_Cat = cardinal_cols + categorical_cols
total_Num = [col for col in num_but_car + numerical_col if col != "SalePrices"]
other = [col for col in df.columns if col not in total_Cat + total_Num]

#Convertimos etiquetas categóticas en números
def labels(dataframe):
    labels_col = [col for col in dataframe.columns if
                  any(value in dataframe[col].unique() for value in
                      ["Ex", "Gd", "GLQ", "Gtl", "Mod", "Sev", "Roll", "CBlock", "Brick Common"])]
    label = LabelEncoder()
    for col in labels_col:
        if any(dataframe[col].isin(["Ex", "Gd", "GLQ", "Gtl", "Mod", "Sev", "Roll", "CBlock", "Brick Common"])):
            dataframe[col] = label.fit_transform(dataframe[col])

labels(df)

##Asignamos unos y ceros a columnas categóricas binarias
def binary_encode(dataframe):
    binary = [col for col in dataframe.columns if
              dataframe[col].dtypes == "object" and dataframe[col].nunique() < 3 and (col != "group")]
    for col in binary:
        if col:
            dataframe[col] = pd.get_dummies(dataframe[col], drop_first=True)


binary_encode(df)

##One hot encoding de variables categóricas
ms_encode = pd.get_dummies(df["MSZoning"], drop_first=True, dtype=int)
df = pd.concat([df, ms_encode], axis=1)
df.drop("MSZoning", axis=1, inplace=True)

lots_encode = pd.get_dummies(df["LotShape"], drop_first=True, dtype=int)
df = pd.concat([df, lots_encode], axis=1)  
df.drop("LotShape", axis=1, inplace=True)

lot_encode = pd.get_dummies(df["LotConfig"], drop_first=True, dtype=int)
df = pd.concat([df, lot_encode], axis=1)  
df.drop("LotConfig", axis=1, inplace=True)

neigh_encode = pd.get_dummies(df["LandContour"], drop_first=True, dtype=int)
df = pd.concat([df, neigh_encode], axis=1)  
df.drop("LandContour", axis=1, inplace=True)

neigh_encode = pd.get_dummies(df["Neighborhood"], drop_first=True, dtype=int)
df = pd.concat([df, neigh_encode], axis=1)  
df.drop("Neighborhood", axis=1, inplace=True)

neigh_encode = pd.get_dummies(df["Condition1"], drop_first=True, dtype=int)
df = pd.concat([df, neigh_encode], axis=1)  
df.drop("Condition1", axis=1, inplace=True)

neigh_encode = pd.get_dummies(df["BldgType"], drop_first=True, dtype=int)
df = pd.concat([df, neigh_encode], axis=1)  
df.drop("BldgType", axis=1, inplace=True)

neigh_encode = pd.get_dummies(df["HouseStyle"], drop_first=True, dtype=int)
df = pd.concat([df, neigh_encode], axis=1)  
df.drop("HouseStyle", axis=1, inplace=True)

neigh_encode = pd.get_dummies(df["RoofStyle"], drop_first=True, dtype=int)
df = pd.concat([df, neigh_encode], axis=1)  
df.drop("RoofStyle", axis=1, inplace=True)

neigh_encode = pd.get_dummies(df["GarageType"], drop_first=True, dtype=int)
df = pd.concat([df, neigh_encode], axis=1)  
df.drop("GarageType", axis=1, inplace=True)

neigh_encode = pd.get_dummies(df["GarageFinish"], drop_first=True, dtype=int)
df = pd.concat([df, neigh_encode], axis=1)  
df.drop("GarageFinish", axis=1, inplace=True)

neigh_encode = pd.get_dummies(df["SaleType"], drop_first=True, dtype=int)
df = pd.concat([df, neigh_encode], axis=1)  
df.drop("SaleType", axis=1, inplace=True)

neigh_encode = pd.get_dummies(df["SaleCondition"], drop_first=True, dtype=int)
df = pd.concat([df, neigh_encode], axis=1)  
df.drop("SaleCondition", axis=1, inplace=True)

##Escalamiento de variables
from sklearn.preprocessing import MinMaxScaler

total_Num = [col for col in num_but_car + numerical_col if col != "SalePrices" and col != "Id"]

min_max_scaler = MinMaxScaler()

# Scaling multiple columns at once
df[total_Num] = min_max_scaler.fit_transform(df[total_Num])

# Scaling individual columns
df["Exterior1st"] = min_max_scaler.fit_transform(df["Exterior1st"].values.reshape(-1, 1))
df["Exterior2nd"] = min_max_scaler.fit_transform(df["Exterior2nd"].values.reshape(-1, 1))
df["ExterQual"] = min_max_scaler.fit_transform(df["ExterQual"].values.reshape(-1, 1))
df["ExterCond"] = min_max_scaler.fit_transform(df["ExterCond"].values.reshape(-1, 1))
df["Foundation"] = min_max_scaler.fit_transform(df["Foundation"].values.reshape(-1, 1))
df["BsmtQual"] = min_max_scaler.fit_transform(df["BsmtQual"].values.reshape(-1, 1))
df["BsmtExposure"] = min_max_scaler.fit_transform(df["BsmtExposure"].values.reshape(-1, 1))
df["BsmtFinType1"] = min_max_scaler.fit_transform(df["BsmtFinType1"].values.reshape(-1, 1))
df["BsmtFinType2"] = min_max_scaler.fit_transform(df["BsmtFinType2"].values.reshape(-1, 1))
df["HeatingQC"] = min_max_scaler.fit_transform(df["HeatingQC"].values.reshape(-1, 1))
df["KitchenQual"] = min_max_scaler.fit_transform(df["KitchenQual"].values.reshape(-1, 1))
df["FireplaceQu"] = min_max_scaler.fit_transform(df["FireplaceQu"].values.reshape(-1,1))

#Guardamos el dataframe limpio
df.to_csv('../data/clean.csv', index=False)
