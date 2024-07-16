from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model # type: ignore
from typing import Tuple, List
import holidays
import pandas as pd
import numpy as np
import os

class Data:

    def __init__(self) -> None:

        self.df = pd.DataFrame()
        self.df_normalized = pd.DataFrame()
        self.min_value_23 = float
        self.max_value_23 = float
        self.min_value_daily = float
        self.max_value_daily = float
        self.min_value_day = float
        self.max_value_day = float

        self.X_train = pd.DataFrame()
        self.X_val = pd.DataFrame()
        self.X_test = pd.DataFrame()


    ################
    # ENEL DATA 
    ################

    def load_ENEL_data(self, folder_path:str, file_name:str) -> None:
        """
        Loads and preprocesses ENEL data from a parquet file.

        This method loads data from the specified parquet file, transforms and normalizes
        the data, and updates the internal DataFrame attributes with the processed data.

        Args:
            folder_path (str): The folder path where the file is located.
            file_name (str): The name of the parquet file to be loaded.
        """

        file_path = os.path.join(folder_path, file_name)
        dataframe = pd.read_parquet(file_path)

        dataframe = self._transform_ENEL_dataframe(dataframe)
        dataframe = self._add_columns_doy_tod(dataframe)
        dataframe = self._add_daily_load_column(dataframe)
        columns_order_df = ["date", "daily_load", "day_of_year", "type_of_day",
                         "0", "1", "2", "3", "4", "5", "6", "7",
                         "8", "9", "10", "11", "12", "13", "14",
                         "15", "16", "17", "18", "19", "20", "21",
                         "22", "23"]
        dataframe = dataframe[columns_order_df]
        self.df = dataframe

        self.df_normalized, self.min_value_23, self.max_value_23 = self.normalize_dataframe(dataframe)
        self.min_value_daily, self.max_value_daily = self._normalize_column(self.df_normalized, "daily_load")
        self.min_value_day, self.max_value_day = self._normalize_column(self.df_normalized, "day_of_year")

        self.df_normalized = self._hot_one_encoding(self.df_normalized, "type_of_day")
        columns_order_df_norm = ["date", "daily_load", "day_of_year", 
                                 "type_of_day_1", "type_of_day_2", "type_of_day_3",
                                "0", "1", "2", "3", "4", "5", "6", "7",
                                "8", "9", "10", "11", "12", "13", "14",
                                "15", "16", "17", "18", "19", "20", "21",
                                "22", "23"]
        
        self.df_normalized = self.df_normalized[columns_order_df_norm]

    def load_UNAL_data(self, folder_path:str, file_name:str) -> None:
        """
        Loads and preprocesses UNAL data from a parquet file.

        This method loads data from the specified CSV file, transforms and normalizes
        the data, and updates the internal DataFrame attributes with the processed data.

        Args:
            folder_path (str): The folder path where the file is located.
            file_name (str): The name of the parquet file to be loaded.
        """
        file_path = os.path.join(folder_path, file_name)
        dataframe = pd.read_csv(file_path, delimiter="\t")
        dataframe = self._transform_UNAL_dataframe(dataframe)
        dataframe = self._add_columns_doy_tod(dataframe)
        dataframe = self._add_daily_load_column(dataframe)
        columns_order_df = ["date", "daily_load", "day_of_year", "type_of_day",
                         "0", "1", "2", "3", "4", "5", "6", "7",
                         "8", "9", "10", "11", "12", "13", "14",
                         "15", "16", "17", "18", "19", "20", "21",
                         "22", "23"]
        dataframe = dataframe[columns_order_df]
        self.df = dataframe

        self.df_normalized, self.min_value_23, self.max_value_23 = self.normalize_dataframe(dataframe)
        self.min_value_daily, self.max_value_daily = self._normalize_column(self.df_normalized, "daily_load")
        self.min_value_day, self.max_value_day = self._normalize_column(self.df_normalized, "day_of_year")

        self.df_normalized = self._hot_one_encoding(self.df_normalized, "type_of_day")
        columns_order_df_norm = ["date", "daily_load", "day_of_year", 
                                 "type_of_day_1", "type_of_day_2", "type_of_day_3",
                                "0", "1", "2", "3", "4", "5", "6", "7",
                                "8", "9", "10", "11", "12", "13", "14",
                                "15", "16", "17", "18", "19", "20", "21",
                                "22", "23"]  
              
        self.df_normalized = self.df_normalized[columns_order_df_norm]

    def load_chulalongkorn_floor_data(self, folder_path:str, files_names:list) -> None:
        """
        Loads and preprocesses Chulalongkorn floor data from CSV files.

        This method loads data from multiple CSV files, concatenates them into a single 
        DataFrame, transforms and normalizes the data, and updates the internal DataFrame 
        attributes with the processed data.

        Args:
            folder_path (str): The folder path where the files are located.
            files_names (list): A list of CSV file names to be loaded.

        Raises:
            Exception: If `files_names` is not a list.
        """
        if not isinstance(files_names, list):
            raise Exception("ERROR: files_names must have to be List type")
        
        dfs=[] # list of Dataframes
        for file in files_names:
            file_path = os.path.join(folder_path,file)
            dfs.append(pd.read_csv(file_path))

        dataframe = pd.concat(dfs, ignore_index=True)
        dataframe = self._transform_chulalongkorn_floor_dataframe(dataframe)
        dataframe = self._add_columns_doy_tod(dataframe)
        dataframe = self._add_daily_load_column(dataframe)
        columns_order_df = ["date", "daily_load", "day_of_year", "type_of_day",
                         "0", "1", "2", "3", "4", "5", "6", "7",
                         "8", "9", "10", "11", "12", "13", "14",
                         "15", "16", "17", "18", "19", "20", "21",
                         "22", "23"]
        dataframe = dataframe[columns_order_df]
        self.df = dataframe

        self.df_normalized, self.min_value_23, self.max_value_23 = self.normalize_dataframe(dataframe)
        self.min_value_daily, self.max_value_daily = self._normalize_column(self.df_normalized, "daily_load")
        self.min_value_day, self.max_value_day = self._normalize_column(self.df_normalized, "day_of_year")

        self.df_normalized = self._hot_one_encoding(self.df_normalized, "type_of_day")
        columns_order_df_norm = ["date", "daily_load", "day_of_year", 
                                 "type_of_day_1", "type_of_day_2", "type_of_day_3",
                                "0", "1", "2", "3", "4", "5", "6", "7",
                                "8", "9", "10", "11", "12", "13", "14",
                                "15", "16", "17", "18", "19", "20", "21",
                                "22", "23"]  
        
        self.df_normalized = self.df_normalized[columns_order_df_norm]


    def normalize_dataframe(self, df:pd.DataFrame, scaling_factor:float = 0.9) -> Tuple[pd.DataFrame, float, float]:
        """
        Normalizes the specified columns of a DataFrame using min-max scaling.

        This method applies min-max normalization to the columns representing the 24 hours 
        in a day within the given DataFrame. The normalization is scaled by a specified 
        scaling factor, adjusting the values to a range between (1 - scaling_factor) / 2 
        and (1 - (1 - scaling_factor) / 2).

        Args:
            df (pd.DataFrame): The DataFrame to be normalized.
            scaling_factor (float): The scaling factor to adjust the normalized range.
                                    Default is 0.9.

        Returns:
            Tuple[pd.DataFrame, float, float]: A tuple containing:
                - df_normalized (pd.DataFrame): The normalized DataFrame.
                - D_min (float): The minimum value found in the columns to be normalized.
                - D_max (float): The maximum value found in the columns to be normalized.
        """
        df_normalized = df.copy()

        columns_to_normalize = [str(i) for i in range(24)]

        # Get the minimum and maximum of all the columns to be normalized
        D_min = df[columns_to_normalize].min().min()
        D_max = df[columns_to_normalize].max().max()

        # Apply normalization to each specified column
        for column in columns_to_normalize:
            df_normalized[column] = scaling_factor * (df[column] - D_min) / (D_max - D_min) + (1 - scaling_factor) / 2

        return df_normalized, D_min, D_max

    def denormalize_dataframe(self, df_normalized, D_min, D_max, scaling_factor=0.9) -> pd.DataFrame:
        """
        Denormalizes a DataFrame's specified columns by applying an inverse of a previously
        applied normalization transformation. The transformation adjusts each value to represent
        its original scale before normalization.

        Parameters:
        df_normalized (pd.DataFrame): The DataFrame with normalized data.
        D_min (float): The minimum value used in the original data before normalization.
        D_max (float): The maximum value used in the original data before normalization.
        scaling_factor (float): The scaling factor used in the normalization process.

        Returns:
        pd.DataFrame: The DataFrame with denormalized data, modified in-place.
        """
        new_df = df_normalized.copy()

        columns_to_denormalize = [str(i) for i in range(24)]  # Adjusted to start from column '0' to '23'
        
        # Apply denormalization to each specified column directly in the original DataFrame
        for column in columns_to_denormalize:
            new_df[column] = (
                (new_df[column] - (1 - scaling_factor) / 2) / scaling_factor) * (D_max - D_min) + D_min

        return new_df

    def set_training_dfs_autoencoder(self, training_percentage:float) -> None:
        """
        Splits input and output data into training, validation, and testing sets.

        Parameters:
        training_percentage (float): Percentage of the total data to allocate to the training set.
        """

        input_dataframe = self.df_normalized
        output_dataframe = self.df_normalized
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_data(input_dataframe, output_dataframe, training_percentage)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

    def get_data_training_autoencoder(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves training, validation, and testing datasets for an autoencoder model from the instance's DataFrame attributes.

        This method extracts the specified hourly columns from the training, validation, and testing DataFrames
        stored in the instance, converts them to NumPy arrays, and returns them for use in training an autoencoder.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Xy_train (np.ndarray): Training data extracted from `self.X_train` using the specified columns.
            - Xy_val (np.ndarray): Validation data extracted from `self.X_val` using the specified columns.
            - Xy_test (np.ndarray): Testing data extracted from `self.X_test` using the specified columns.

        The columns used are numeric representations of hours (0 through 23)
        """
        columns_names_hours = [str(i) for i in range(24)]
        Xy_train = self.X_train[columns_names_hours].values
        Xy_val = self.X_val[columns_names_hours].values
        Xy_test = self.X_test[columns_names_hours].values

        return Xy_train, Xy_val, Xy_test 

    def get_data_training_ANN_hidden_varibles(self, encoder_model: Model, training_percentage: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates training, validation, and testing datasets for an ANN model that uses encoded hidden variables.

        Parameters:
        encoder_model (Model): A trained encoder model that transforms input data into a hidden representation.
        training_percentage (float): The fraction of data to be used for training. The rest is split evenly between validation and testing.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_train (np.ndarray): Input features for training the ANN.
            - y_train (np.ndarray): Output targets for training the ANN.
            - X_val (np.ndarray): Input features for validating the ANN.
            - y_val (np.ndarray): Output targets for validating the ANN.
            - X_test (np.ndarray): Input features for testing the ANN.
            - y_test (np.ndarray): Output targets for testing the ANN.
        """
        # Encoder Model only has 24 input parameters
        columns_names_hours = [str(i) for i in range(24)]
        predict_hidden_varibles = encoder_model.predict(self.X_test[columns_names_hours].values)

        # Output varibles for "ANN_hidden_varibles" are the hidden varibles
        columns_names_hidden_varibles = [f"hidden_varible_{x}" for x in range(predict_hidden_varibles.shape[1])]
        df_output = pd.DataFrame(data=predict_hidden_varibles, columns=columns_names_hidden_varibles)
        
        # Input varibles for "ANN_hidden_varibles" are "day_of_year", "type_of_day_1", "type_of_day_2", "type_of_day_3"
        columns_names_others = ["day_of_year", "type_of_day_1", "type_of_day_2", "type_of_day_3"]
        df_input = self.X_test[columns_names_others]

        # Split Dataframes
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_data(input_dataframe = df_input, output_dataframe = df_output, training_percentage = training_percentage)
        
        return X_train.values, y_train.values, X_val.values, y_val.values, X_test.values, y_test.values
    
    def get_data_training_ANN_daily_load(self,  encoder_model: Model, training_percentage: float)  -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # Encoder Model only has 24 input parameters
        columns_names_hours = [str(i) for i in range(24)]
        predict_hidden_varibles = encoder_model.predict(self.X_test[columns_names_hours].values)

        # "date", "daily_load", "day_of_year", "type_of_day_1", "type_of_day_2", "type_of_day_3"
        
        org_df = self.X_test
        columns_names_hidden_varibles = [f"hidden_varible_{x}" for x in range(predict_hidden_varibles.shape[1])]
        hidden_varibles_df = pd.DataFrame(data=predict_hidden_varibles, columns=columns_names_hidden_varibles)

        if org_df.shape[0] != hidden_varibles_df.shape[0]:
            raise Exception(f"There is a problem with size of hidden_varibles_predict and X_test_df")
        
        else: 
            org_df.reset_index(drop=True, inplace=True)
            hidden_varibles_df.reset_index(drop=True, inplace=True)
            new_df =  pd.concat([org_df, hidden_varibles_df], axis=1)

            columns_names_input = ["day_of_year", "type_of_day_1", "type_of_day_2", "type_of_day_3"] + columns_names_hidden_varibles
            input_df = new_df[columns_names_input]

            output_df = new_df[["daily_load"]]

            X_train, y_train, X_val, y_val, X_test, y_test = self._split_data(input_dataframe = input_df, output_dataframe = output_df, training_percentage = training_percentage)
        
        return X_train.values, y_train.values, X_val.values, y_val.values, X_test.values, y_test.values

    def _split_data(self, input_dataframe: pd.DataFrame, output_dataframe:pd.DataFrame, training_percentage:float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits input and output data into training, validation, and testing sets based on the specified training percentage.

        Parameters:
        input_dataframe (pd.DataFrame): DataFrame containing the input features.
        output_dataframe (pd.DataFrame): DataFrame containing the output targets.
        training_percentage (float): Percentage of the total data to allocate to the training set.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - X_train (pd.DataFrame): Training data features.
        - y_train (pd.DataFrame): Training data targets.
        - X_val (pd.DataFrame): Validation data features.
        - y_val (pd.DataFrame): Validation data targets.
        - X_test (pd.DataFrame): Testing data features.
        - y_test (pd.DataFrame): Testing data targets.

        The function first divides the data into training and temporary sets (the latter containing both validation and test data).
        It then splits the temporary set evenly into validation and test sets. The training percentage determines how much data is used for training
        while the remainder is equally split between validation and testing.
        """

        X_train, X_temp, y_train, y_temp = train_test_split(
            input_dataframe, output_dataframe, train_size=training_percentage)
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5)
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _transform_ENEL_dataframe(self, original_df) -> pd.DataFrame:
        """
        Transforms a DataFrame by filtering dates that have exactly 24 hours of measurements
        and restructuring the DataFrame so that each row represents a full day of measurements.

        Parameters:
        original_df (pd.DataFrame): The original DataFrame containing measurements distributed
                                    across multiple rows by date, including columns for 'date',
                                    'month', 'day', 'year', 'day_of_week', 'type_of_day', 'hour', 
                                    and 'measurement_'.

        Returns:
        pd.DataFrame: A new DataFrame where each row corresponds to a complete day with 24 hours
                    of measurements. Each row includes the date, day details, and measurements
                    for each hour of the day from 0 to 23. If any hour is missing, its value
                    will be None and an error message will be printed
        """

        # Filter days that do not have 24 hours of measurements
        counts_per_day = original_df.groupby('fecha').size()
        valid_dates = counts_per_day[counts_per_day == 24].index
        filtered_df = original_df[original_df['fecha'].isin(valid_dates)]

        new_columns = ['date', 'month', 'day', 'year', 'day_week', 'day_type'] + [str(i) for i in range(24)]
        rows_list = []  # List to store rows as dictionaries

        for date in valid_dates:
            day_data = filtered_df[filtered_df['fecha'] == date]
            new_row = {
                'date': date,
                'month': day_data['mes'].iloc[0],
                'day': day_data['dia'].iloc[0],
                'year': day_data['anio'].iloc[0],
                'day_week': day_data['dia_semana'].iloc[0],
                'day_type': day_data['tipo_dia'].iloc[0],
            }

            for i in range(24):
                if i in day_data['hora'].values:
                    new_row[str(i)] = day_data[day_data['hora'] == i]['medicion'].iloc[0]
                else:
                    new_row[str(i)] = None
                    print(f"error en el dia {date} y hora {i}")

            rows_list.append(new_row)

        new_df = pd.DataFrame(rows_list, columns=new_columns)  # Create the new DataFrame from the dictionary list
        return new_df

    def _transform_UNAL_dataframe(self, df):
        
        # Verificar si el DataFrame tiene las columnas esperadas
        if not all(col in df.columns for col in ['FECHA', 'HORA', 'kW_Tot']):
            raise ValueError("El DataFrame de entrada debe tener las columnas 'FECHA', 'HORA' y 'kW_Tot'")
        
        # Convertir las columnas FECHA y HORA a datetime
        df['FECHA_HORA'] = pd.to_datetime(df['FECHA'] + ' ' + df['HORA'])
        df['FECHA'] = pd.to_datetime(df['FECHA'])
        
        # Crear una columna para el día
        df['date'] = df['FECHA_HORA'].dt.date
        
        # Crear una columna para la hora
        df['hour'] = df['FECHA_HORA'].dt.hour
        
        # Verificar si cada día tiene 24 mediciones
        daily_counts = df['date'].value_counts()
        valid_dates = daily_counts[daily_counts == 24].index
        
        # Filtrar el DataFrame para mantener solo los días con 24 mediciones
        df = df[df['date'].isin(valid_dates)]
        
        # Pivotar el DataFrame
        df_pivot = df.pivot_table(index='date', columns='hour', values='kW_Tot')
        
        # Asegurarse de que todas las horas de 0 a 23 estén presentes en formato string
        for hour in range(24):
            if hour not in df_pivot.columns:
                df_pivot[hour] = np.nan
                print(f"WARNING: Emtpy value found hour {hour}")
        
        # Renombrar las columnas de horas a string si no están ya en formato string
        df_pivot.columns = df_pivot.columns.astype(str)
        
        # Ordenar las columnas de las horas
        df_pivot = df_pivot.sort_index(axis=1)
        
        # Convertir el índice (date) en columna
        df_pivot.reset_index(inplace=True)
        
        # Determinar el tipo de día
        co_holidays = holidays.Colombia()
        df_pivot['day_type'] = df_pivot['date'].apply(
            lambda x: 'domingo_festivo' if x in co_holidays or x.weekday() == 6 else ('sabado' if x.weekday() == 5 else 'entre_semana')
        )
        
        # Reordenar las columnas
        df_pivot = df_pivot[['date', 'day_type'] + [str(hour) for hour in range(24)]]
        
        return df_pivot

    def _transform_chulalongkorn_floor_dataframe(self, df):
        
        if 'Date' not in df.columns:
            raise ValueError("El DataFrame de entrada debe tener la columna 'Date'")
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        df['date'] = df['Date'].dt.date
        df['hour'] = df['Date'].dt.hour
        
        # Identificar columnas que terminan con "(kW)"
        kw_columns = [col for col in df.columns if col.endswith('(kW)')]
        
        # Filtrar las columnas necesarias y sumar los valores por día y hora
        df['sum_kW'] = df[kw_columns].sum(axis=1)
        
        # Pivotar el DataFrame
        df_pivot = df.pivot_table(index='date', columns='hour', values='sum_kW', aggfunc='sum')
        
        # Asegurarse de que todas las horas de 0 a 23 estén presentes
        for hour in range(24):
            if hour not in df_pivot.columns:
                df_pivot[hour] = np.nan
                print(f"WARNING: Emtpy value found hour {hour}")
        
        # Renombrar las columnas de horas a string si no están ya en formato string
        df_pivot.columns = df_pivot.columns.astype(str)
        
        # Ordenar las columnas de las horas
        df_pivot = df_pivot.sort_index(axis=1)
        
        # Convertir el índice (date) en columna
        df_pivot.reset_index(inplace=True)
        
        # Determinar el tipo de día
        th_holidays = holidays.TH()
        df_pivot['day_type'] = df_pivot['date'].apply(
            lambda x: 'domingo_festivo' if x in th_holidays or x.weekday() == 6 else ('sabado' if x.weekday() == 5 else 'entre_semana')
        )
        
        # Reordenar las columnas
        df_pivot = df_pivot[['date', 'day_type'] + [str(hour) for hour in range(24)]]
        
        return df_pivot

    def _add_columns_doy_tod(self, dataframe) -> pd.DataFrame:
        """
        Adds two new columns to a DataFrame: 'day_of_year' and 'type_of_day'.

        Parameters:
        dataframe (pd.DataFrame): The original DataFrame containing the columns 'date' and 'day_type'.

        Returns:
        pd.DataFrame: The modified DataFrame with two new columns:
        - 'day_of_year': The day of the year based on the 'date' column, with values from 1 to 365 (or 366 in leap years).
        - 'type_of_day': A numerical category based on the 'day_type' column, where 1 represents weekdays,
        2 represents Saturdays, and 3 represents Sundays or holidays.
        """
        # Convert 'date' column to datetime if it is not already
        if not pd.api.types.is_datetime64_any_dtype(dataframe['date']):
            dataframe['date'] = pd.to_datetime(dataframe['date'])

        # Add 'day_of_year' column
        dataframe['day_of_year'] = dataframe['date'].dt.dayofyear

        # Define a mapping of day types to numeric values
        type_of_day_mapping = {
            'entre_semana': 1,
            'sabado': 2,
            'domingo_festivo': 3
        }

        # Apply mapping to create 'type_of_day' column
        dataframe['type_of_day'] = dataframe['day_type'].map(type_of_day_mapping)

        return dataframe

    def _add_daily_load_column(self, dataframe) -> pd.DataFrame:

        # Calculate 'daily_load' column as the sum of hourly columns
        columns_to_sum = [str(i) for i in range(24)]
        dataframe['daily_load'] = dataframe[columns_to_sum].sum(axis=1)

        return dataframe

    def _normalize_column(self, dataframe: pd.DataFrame, column_name:str) -> Tuple[float, float]:
            """
            Normalizes a specified column in a DataFrame.

            Parameters:
            dataframe (pd.DataFrame): The DataFrame containing the column to be normalized.
            column_name (str): The name of the column to normalize.

            Returns:
            None
            """

            if column_name not in dataframe.columns:
                raise ValueError(f"The column '{column_name}' is not in the DataFrame")
            
            min_val = dataframe[column_name].min()
            max_val = dataframe[column_name].max()

            dataframe[column_name] = (dataframe[column_name] - min_val) / (max_val - min_val)

            return min_val, max_val

    def _denormalize_column(self, dataframe: pd.DataFrame, column_name: str, min_val: float, max_val: float) -> None:
        """
        Denormalizes a specified column in a DataFrame using the provided minimum and maximum values.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the column to be denormalized.
        column_name (str): The name of the column to denormalize.
        min_val (float): The minimum value of the original data before normalization.
        max_val (float): The maximum value of the original data before normalization.

        Returns:
        None: The column within the DataFrame is modified in-place.
        """

        if column_name not in dataframe.columns:
            raise ValueError(f"The column '{column_name}' is not in the DataFrame")
        else:
        # Perform the denormalization using the original min and max values
            dataframe[column_name] = dataframe[column_name] * (max_val - min_val) + min_val

    def _hot_one_encoding(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified column in the DataFrame and removes the original column.

        Parameters:
        df (pd.DataFrame): The DataFrame to modify.
        column_name (str): The name of the column to apply one-hot encoding.

        Returns:
        pd.DataFrame: The DataFrame with the one-hot encoded column and the original column removed.

        Raises:
        ValueError: If the specified column does not exist in the DataFrame.
        """
        if column_name not in df.columns:
            raise ValueError(f"The column '{column_name}' is not in the DataFrame")
        
        # Perform one-hot encoding
        dummies = pd.get_dummies(df[column_name], prefix=column_name)

        dummies = dummies.astype('float32')
        
        # Concatenate the new columns with the original DataFrame
        df = pd.concat([df, dummies], axis=1)
        
        # Remove the original column
        df.drop(column_name, axis=1, inplace=True)

        return df
