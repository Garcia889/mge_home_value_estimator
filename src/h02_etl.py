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

#Leemos y combinamos datos de entrenamiento y prueba
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

