"""
Este archivo contiene el código para realizar la etapa de extracción, transformación y carga (ETL) de los datos.
Realiza las siguientes tareas:

1. Reads raw homes data from a CSV file
2. Separa las variables categóricas y numéricas.
3. Maneja los valores nulos en las variables categóricas y numéricas.
4. Elimina variables categóricas dominantes y variables numéricas con baja correlación.
5. Agrega nuevas variables al conjunto de datos.
6. Realiza codificación de etiquetas y codificación binaria en variables categóricas.
7. Realiza codificación one-hot en variables categóricas seleccionadas.
8. Escala las variables numéricas utilizando MinMaxScaler.
9. Guarda el conjunto de datos limpio en un archivo CSV (clean.csv).
"""

##Bibliotecas
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class dataPreprocessor:
    def __init__(self, homes_input, homes_output, logs_output):
        self.homes_input = homes_input
        self.homes_output = homes_output
        self.logs_output = logs_output
        ##Logging configuration
        logging.basicConfig(
            filename=self.logs_output,
            level=logging.INFO,
            filemode='a',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ##Pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.width', 500)

    def load_homes_csv(self):
            """
            Loads a CSV file containing homes data.

            Returns:
                pandas.DataFrame: The loaded CSV data as a DataFrame.
            """
            return pd.read_csv(self.homes_input)
    
    ##Separate categorical and numerical variables
    def separate_variables(self, df):
        """
        Separates the variables in the given DataFrame into categorical, numerical, and other variables.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the variables.

        Returns:
        tuple: A tuple containing three lists: 
            total_Cat (list): The list of categorical variables.
            total_Num (list): The list of numerical variables.
            other (list): The list of other variables.
        """
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

        logging.info('Primer conteo de variables')
        logging.info(f"Categóricas: {len(total_Cat)} {total_Cat}")
        logging.info(f"Numéricas: {len(total_Num)} {total_Num}")
        logging.info(f"Otras variables: {len(other)} {other}")

        return total_Cat, total_Num, other
    

    ##In categorical variables, we check for null values and 
    ##for this exercise, we assign them zero.
    def nan_detected(self, dataframe, col_names):
        nan_values = dataframe[col_names].isnull().sum()
        nan_ratio = nan_values / len(dataframe[col_names])
        logging.info("Nan values detected in categorical variables: ")
        logging.info(f"Nan values: {nan_values}\n Nan ratio: {nan_ratio}")


    def nan_fixcat(self, dataframe, col_names):
        for col in col_names:
            if dataframe[col].isnull().sum() > 0:
                dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)
        logging.info("NaN values fixed successfully.")
        return dataframe

    ##In numerical variables, checks for null values and assign them the mean if they exist
    ##logs the number of fixes for each numerical variable
    def nan_fixnum(self, dataframe, col_names):
        fixes = dataframe[col_names].fillna(dataframe[col_names].median(), inplace=True)
        if fixes is not None and fixes > 0:
            logging.info(f'Numeric nulls {col_names} {fixes}')


    ##Removes categorical variables where more than 90% of their records have the same value
    def dominant_values(self, dataframe, col):
        dominant_ratio = (dataframe[col].value_counts() / len(dataframe[col])) * 100

        threshold = 90

        if dominant_ratio.max() > threshold:
            dataframe.drop(col, axis=1, inplace=True)
            return f"{col} 90% dominant value, dropped successfully."

    ##Agregamos nuevas variables como Edad de la casa o número total de baños
    def new_features(self, dataframe):
        dataframe['HouseAge'] = dataframe['YrSold'] - dataframe['YearBuilt']
        dataframe['OverallQualityCondition'] = dataframe['OverallQual'] + dataframe['OverallCond']
        dataframe['TotalBathrooms'] = dataframe['BsmtFullBath'] + dataframe['FullBath']
        logging.info('New features added successfully.')
        return dataframe

    def new_variables(self, df):
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
        return total_Cat, total_Num, other, numerical_col, num_but_car

    #Convert categorical labels into numbers
    def labels(self, dataframe):
        labels_col = [col for col in dataframe.columns if
                    any(value in dataframe[col].unique() for value in
                        ["Ex", "Gd", "GLQ", "Gtl", "Mod", "Sev", "Roll", "CBlock", "Brick Common"])]
        label = LabelEncoder()
        for col in labels_col:
            if any(dataframe[col].isin(["Ex", "Gd", "GLQ", "Gtl", "Mod", "Sev", "Roll", "CBlock", "Brick Common"])):
                dataframe[col] = label.fit_transform(dataframe[col])
        logging.info('Labels encoded successfully.')

    ##Assign ones and zeros to binary categorical columns
    def binary_encode(self, dataframe):
        binary = [col for col in dataframe.columns if
                dataframe[col].dtypes == "object" and dataframe[col].nunique() < 3 and (col != "group")]
        for col in binary:
            if col:
                dataframe[col] = pd.get_dummies(dataframe[col], drop_first=True)
        logging.info('Binary encoded successfully.')

    ##One hot encoding of categorical variables
    def one_hot_encoding(self, dataframe):
        df = dataframe.copy()

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

        logging.info('One hot encoding completed successfully.')

        return df

    def scale_variables(self, dataframe, numerical_col, num_but_car):
        df = dataframe.copy()
        ##Escalamiento de variables
        total_Num = [col for col in num_but_car + numerical_col if col != "SalePrices" and col != "Id"]

        min_max_scaler = MinMaxScaler()

        ##Escalamiento de múltiples variables en un sólo paso
        df[total_Num] = min_max_scaler.fit_transform(df[total_Num])

        ##Escalamiento de columnas individuales
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
        logging.info('Variables successfully scaled.')
        return df
    
    #Guarda el dataframe limpio
    def save_clean_csv(self, df, output):
        df.to_csv(output, index=False)
        logging.info(f'Archivo {output} guardado exitosamente.')
        logging.info('ETL finalizado exitosamente!')

    def procesar_csv_homes(self):
        df = self.load_homes_csv()
        df = df.drop("Id", axis=1)
        total_Cat, total_Num, other = self.separate_variables(df)
        self.nan_detected(df, total_Cat)
        df = self.nan_fixcat(df, total_Cat)
        for col in total_Num:
            self.nan_fixnum(df, col)
        for col in total_Cat:
            has_dominant = self.dominant_values(df, col)
            if has_dominant is not None:
                logging.info(f'{has_dominant}')
        df = self.new_features(df)
        total_Cat, total_Num, other, numerical_col, num_but_car = self.new_variables(df)
        self.labels(df)
        self.binary_encode(df)
        df = self.one_hot_encoding(df)
        df = self.scale_variables(df, numerical_col, num_but_car)
        self.save_clean_csv(df, self.homes_output)
 

if __name__ == "__main__":
    homes_input = "../data/test.csv"
    homes_output = "../data/clean_test.csv"
    procesador = dataPreprocessor(homes_input, homes_output)
    procesador.procesar_csv_homes()

    #df_procesado = procesador.procesar_csv_homes()
    #print("DataFrame procesado:")
    #print(df_procesado)
