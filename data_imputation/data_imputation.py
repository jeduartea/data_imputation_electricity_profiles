from tensorflow import convert_to_tensor
from numpy import array, ndarray, random
from pandas import DataFrame
from typing import Tuple

from data_imputation.preprocessing import Data
from data_imputation.autoencoder import Autoencoder
from data_imputation.artificial_neural_networks import ANN


class DataImputation:

    def __init__(self) -> None:
        # Data parameters
        self.data = Data()
        self.data_folder_path = ""
        self.data_file_name = ""
        self.df_type = ""
        self.data_name = ""
        self.training_percentage = 0.6

        # Autoencoder parameters
        self.model_autoencoder = None
        self.n_input_parameters_AE = 24
        self.n_hidden_layers_AE = 10
        self.n_neuron_hidden_layers_AE = 12
        self.latent_dim_AE = 5
        self.epochs_AE = 300
        self.batch_size_AE = 10

        # ANN hidden varibles (HV) parameters
        self.model_ANN_HV = None
        self.n_hidden_layers_ANN_HV = 10
        self.n_neuron_hidden_layers_ANN_HV = 25
        self.epochs_HV = 100
        self.batch_size_HV = 10

        # ANN daily load (DL) parameters 
        self.model_ANN_DL = None
        self. n_hidden_layers_DL = 10
        self.n_neuron_hidden_layers_DL = 25
        self.epochs_DL = 50 
        self.batch_size_DL = 10

        self._print_msg()

    def set_data_parameters(self, data_folder_path:str, data_file_name:str|list, df_type:str, training_percentage:float) -> None:
        
        self.data_folder_path = data_folder_path
        self.data_file_name = data_file_name
        self.df_type = df_type
        self.data_name = df_type
        self.training_percentage = training_percentage

        if isinstance(data_file_name, list):
            print("WARINING: The varible 'data_file_name' only can be List type when data is from Chulalongkorn University")

    def set_autoencoder_parameters(self, n_input_parameters_AE : int, n_hidden_layers_AE: int, n_neuron_hidden_layers_AE :int, latent_dim_AE : int, epochs_AE : int, batch_size_AE : int) -> None:
        
        self.n_input_parameters_AE = n_input_parameters_AE
        self.n_hidden_layers_AE = n_hidden_layers_AE
        self.n_neuron_hidden_layers_AE = n_neuron_hidden_layers_AE
        self.latent_dim_AE = latent_dim_AE
        self.epochs_AE = epochs_AE
        self.batch_size_AE = batch_size_AE

    def set_ANN_hidden_varibles_parameters(self, n_hidden_layers_ANN_HV : int, n_neuron_hidden_layers_ANN_HV : int, epochs_HV : int, batch_size_HV : int):

        self.n_hidden_layers_ANN_HV = n_hidden_layers_ANN_HV
        self.n_neuron_hidden_layers_ANN_HV = n_neuron_hidden_layers_ANN_HV
        self.epochs_HV = epochs_HV
        self.batch_size_HV = batch_size_HV

    def set_ANN_daily_load_parameters(self, n_hidden_layers_DL : int , n_neuron_hidden_layers_DL : int, epochs_DL : int, batch_size_DL : int) -> None:
        
        self. n_hidden_layers_DL = n_hidden_layers_DL
        self.n_neuron_hidden_layers_DL = n_neuron_hidden_layers_DL
        self.epochs_DL = epochs_DL
        self.batch_size_DL = batch_size_DL

    def load_data_in_models(self) -> None:
        # Load data from files
        if self.df_type.lower() == "enel":
            self.data.load_ENEL_data(self.data_folder_path, self.data_file_name)
        elif self.df_type.lower() == "unal":
            self.data.load_UNAL_data(self.data_folder_path, self.data_file_name)
        elif self.df_type.lower() == "chulalongkorn":
            self.data.load_chulalongkorn_floor_data(self.data_folder_path, self.data_file_name)
        else:
            raise Exception("ERROR: please select one of these dataframe type: \n - enel\n - unal\n - chulalongkorn")
       
    def training_AE(self) -> None:
        """
        Trains the Autoencoder model using the specified parameters and training data.

        This method initializes the Autoencoder model with the given configuration 
        parameters, sets the training data within the data object, retrieves the 
        training, validation, and test datasets, and then proceeds to train the 
        Autoencoder model. Finally, it evaluates the trained model using the test 
        data and prints the resulting metrics.

        Attributes:
            model_autoencoder: An instance of the Autoencoder class initialized with
                            the specified input parameters, hidden layers, neurons,
                            and latent dimension.
            data: An instance containing the training dataframes for the Autoencoder.
            training_percentage: The percentage of data to be used for training.
            epochs_AE: The number of epochs for training the Autoencoder.
            batch_size_AE: The batch size for training the Autoencoder
        """
        self.model_autoencoder = Autoencoder(n_input_parameters = self.n_input_parameters_AE,
                                                n_hidden_layers = self.n_hidden_layers_AE,
                                                n_neuron_hidden_layers = self.n_neuron_hidden_layers_AE, 
                                                latent_dim = self.latent_dim_AE,
                                                name_autoencoder = f"{self.data_name}_autoencoder")


        # Set training dataframes in object "data"
        self.data.set_training_dfs_autoencoder(self.training_percentage)

        # Get training values for autoencoder
        Xy_train_AE, Xy_val_AE, Xy_test_AE =  self.data.get_data_training_autoencoder()

        # Training autoencoder model
        print("\n")
        print("**"*20)
        print(" TRAININING AUTOENCODER")
        print("**"*20)
        self.model_autoencoder.training(Xy_train_AE, Xy_val_AE, self.epochs_AE, self.batch_size_AE)

        print("\n")
        print(" TRAINING MEASURE PARAMETRES WITH TEST DATA:")
        print("       AUTOENCODER")
        metrics_AE = self.model_autoencoder.test(Xy_test_AE)
        print(metrics_AE)
    
    def training_ANN_HV(self)-> None:
        """
        Trains the Artificial Neural Network (ANN) for hidden variable prediction.

        This method retrieves the training, validation, and test datasets for the ANN 
        model using the encoded data from the Autoencoder. It then initializes the ANN 
        model with the specified configuration parameters and trains it using the 
        training and validation datasets. Finally, it evaluates the trained ANN model 
        using the test data and prints the resulting metrics.

        Attributes:
            model_autoencoder.encoder: The encoder part of the trained Autoencoder model.
            data: An instance containing the training data for the ANN hidden variables.
            training_percentage: The percentage of data to be used for training.
            epochs_HV: The number of epochs for training the ANN.
            batch_size_HV: The batch size for training the ANN.
            n_hidden_layers_ANN_HV: The number of hidden layers in the ANN.
            n_neuron_hidden_layers_ANN_HV: The number of neurons per hidden layer in the ANN.
            data_name: A name identifier for the dataset.
        """
        ########################################
        # Hidden variable prediction
        ########################################

        # training data for ANN hidden varibles
        (
            X_train_ANN_hv,
            y_train_ANN_hv,
            X_val_ANN_hv,
            y_val_ANN_hv,
            X_test_ANN_hv,
            y_test_ANN_hv
        ) = self.data.get_data_training_ANN_hidden_varibles(
                encoder_model=self.model_autoencoder.encoder,
                training_percentage=self.training_percentage)

        # ANN hidden varibles model definition
        self.model_ANN_HV = ANN(n_input_parameters = X_train_ANN_hv.shape[1], # input data size
                                    n_output_parameters= y_train_ANN_hv.shape[1], # output data size
                                    n_hidden_layers = self.n_hidden_layers_ANN_HV, 
                                    n_neuron_hidden_layers = self.n_neuron_hidden_layers_ANN_HV,
                                    name_ann = f"{self.data_name}_ANN_hidden_varibles",
                                    output_layer_activation_function="sigmoid")

        print("\n")
        print("**"*20)
        print(" TRAININING ARTIFICIAL NEURAL NETWORK")
        print("         HIDDEN VARIBLES")
        print("**"*20)

        self.model_ANN_HV.training(X_train_ANN_hv, y_train_ANN_hv, X_val_ANN_hv,
                                  y_val_ANN_hv, epochs = self.epochs_HV, batch_size = self.batch_size_HV)
        print("\n")
        print(" TRAINING MEASURE PARAMETRES WITH TEST DATA:")
        print("    ANN HIDDEN VARIBLES")
        metrics_HV = self.model_ANN_HV.test(X_test_ANN_hv, y_test_ANN_hv)
        print(metrics_HV)

    def training_ANN_DL(self) -> None:
        """
        Trains the Artificial Neural Network (ANN) for daily load prediction.

        This method retrieves the training, validation, and test datasets for the ANN 
        model using the encoded data from the Autoencoder. It then initializes the ANN 
        model with the specified configuration parameters and trains it using the 
        training and validation datasets. Finally, it evaluates the trained ANN model 
        using the test data and prints the resulting metrics.

        Attributes:
            model_autoencoder.encoder: The encoder part of the trained Autoencoder model.
            data: An instance containing the training data for the ANN daily load prediction.
            training_percentage: The percentage of data to be used for training.
            epochs_DL: The number of epochs for training the ANN.
            batch_size_DL: The batch size for training the ANN.
            n_hidden_layers_DL: The number of hidden layers in the ANN.
            n_neuron_hidden_layers_DL: The number of neurons per hidden layer in the ANN.
            data_name: A name identifier for the dataset.
        """
        ########################################
        # Daily load prediction
        ########################################
        (
            X_train_ANN_dl,
            y_train_ANN_dl,
            X_val_ANN_dl,
            y_val_ANN_dl,
            X_test_ANN_dl,
            y_test_ANN_dl
        ) = self.data.get_data_training_ANN_daily_load(
                encoder_model=self.model_autoencoder.encoder,
                training_percentage=self.training_percentage)

        self.model_ANN_DL = ANN(n_input_parameters = X_train_ANN_dl.shape[1], # input data size
                                    n_output_parameters= y_train_ANN_dl.shape[1], # output data size
                                    n_hidden_layers = self.n_hidden_layers_DL, 
                                    n_neuron_hidden_layers = self.n_neuron_hidden_layers_DL,
                                    name_ann=f"{self.data_name}_ANN_daily_load",
                                    output_layer_activation_function="sigmoid")
        
        # ANN daily_load model training
        print("\n")
        print("**"*20)
        print(" TRAININING ARTIFICIAL NEURAL NETWORK")
        print("         DAILY LOAD")
        print("**"*20)

        self.model_ANN_DL.training(X_train_ANN_dl, y_train_ANN_dl, X_val_ANN_dl,
                                        y_val_ANN_dl, 
                                        epochs = self.epochs_DL, 
                                        batch_size = self.batch_size_DL)
        print("\n")
        print(" TRAINING MEASURE PARAMETRES  WITH TEST DATA:")
        print("     ANN DAILY LOAD")
        metrics_DL = self.model_ANN_DL.test(X_test_ANN_dl, y_test_ANN_dl)
        print(metrics_DL)

    def training_all_models(self) -> None:
        self.training_AE()
        self.training_ANN_HV()
        self.training_ANN_DL()
        
    def predict_profile(self, day_of_year : int, type_of_day : int, decimal_randomness : int) -> DataFrame:

        day_of_year_test, type_of_day_1_test, type_of_day_2_test, type_of_day_3_test = self._input_varibles_trasformation(day_of_year, type_of_day)

        ########################################
        # Hidden variable prediction
        ########################################

        # input varibles [[day_of_year_normarmalice, type_of_day_1, type_of_day_2, type_of_day_3 ]]

        
        input_varibles = convert_to_tensor(array([[day_of_year_test, type_of_day_1_test, type_of_day_2_test, type_of_day_3_test]]))
        predicted_hidden_values = convert_to_tensor(self.model_ANN_HV.ann_model.predict(input_varibles))

        # add randomness
        predicted_hidden_values_noise = self._add_randomness(predicted_hidden_values, decimal_randomness)

        ########################################
        # Shape predict
        ########################################

        shape_predict = self.model_autoencoder.decoder.predict(predicted_hidden_values_noise)

        # add randomness
        shape_predict_noise = self._add_randomness(shape_predict, decimal_randomness)

        # Denormalize
        predict_df = DataFrame(shape_predict_noise, columns = [str(hour) for hour in range(24)])
        predict_df = self.data.denormalize_dataframe(predict_df, self.data.min_value_23, self.data.max_value_23, scaling_factor=0.9)
        

        ########################################
        # Daily load prediction
        ########################################

        # input varibles  [[day_of_year type_of_day_1 type_of_day_2 type_of_day_3 hidden_varible_1 ... hidden_varible_X]]

        input_varibles_daily_load_test = [day_of_year_test, type_of_day_1_test, type_of_day_2_test, type_of_day_3_test] + [hidden_varible for hidden_varible in predicted_hidden_values_noise[0,:]]

        input_varibles_daily_load_test = convert_to_tensor(array([input_varibles_daily_load_test]).astype('float32'))

        predicted_daily_load = self.model_ANN_DL.ann_model.predict(input_varibles_daily_load_test)

        print("\nPredicted hidden varibles")
        print(f"values are           : {predicted_hidden_values[0, :]}")
        print(f"values with noise are: {predicted_hidden_values_noise[0, :]}\n")

        print("Predicted shape or profile")
        print(f"value are            : {shape_predict}")
        print(f"values with noise are: {shape_predict_noise}")

        print("Predicted daily load")
        print(f"value is: {predicted_daily_load}")

        return predict_df

    def get_test_dataframe(self) -> DataFrame:
        
        dataframe = self.data.denormalize_dataframe(df_normalized = self.data.X_test, 
                                          D_min = self.data.min_value_23, 
                                          D_max = self.data.max_value_23,
                                          scaling_factor=0.9)

        self.data._denormalize_column(dataframe = dataframe, column_name = "daily_load",
                                         min_val = self.data.min_value_daily, 
                                         max_val = self.data.max_value_daily) 

        self.data._denormalize_column(dataframe = dataframe, column_name = "day_of_year",
                                         min_val = self.data.min_value_day, 
                                         max_val = self.data.max_value_day)

        return dataframe

    def _input_varibles_trasformation(self, day_of_year, type_of_day) -> Tuple[float, float, float, float]:

        # hot one enconding
        type_of_day_1 = 0
        type_of_day_2 = 0
        type_of_day_3 = 0

        if type_of_day == 1:
            type_of_day_1 = 1
        elif type_of_day == 2:
            type_of_day_2 = 1
        elif type_of_day == 3:
            type_of_day_3 = 1

        # normalization
        X_std = (day_of_year - 1) / (366 - 1)
        X_scaled = X_std * (1 - 0) + 0

        day_of_year = X_scaled

        return float(day_of_year), float(type_of_day_1), float(type_of_day_2), float(type_of_day_3)

    def _add_randomness(self, values: ndarray, decimal_randomness: int) -> ndarray:
        """
        Adds random noise to the numbers in a NumPy array starting from a specified decimal place.

        Parameters:
        values (ndarray): The array of values to which randomness will be added.
        decimal_randomness (int): The decimal place from which random noise should start.

        Returns:
        ndarray: The array with random noise added.

        Example:
        add_randomness(array([1.2345, 2.3456]), 3) could return np.array([1.235, 2.346]) where
        randomness is added starting from the third decimal place.
        """
        # Calculate the factor to scale the random noise
        noise_scale = 10 ** (-decimal_randomness)

        # Generate random noise and scale it to the desired decimal place
        random_noise = (random.rand(*values.shape) - 0.5) * 2 * noise_scale

        # Add the noise to the original values
        new_values = values + random_noise

        return new_values

    def _print_msg(self) -> None:

        print("You have created an object of the DataImputation class.")
        print("This class facilitates the generation of daily energy consumption profiles. To use this class effectively, please follow these steps:\n")
        print("1. Set up the class parameters by calling the following methods:")
        print("   a. set_data_parameters()")
        print("   b. set_autoencoder_parameters()")
        print("   c. set_ANN_hidden_variables_parameters()")
        print("   d. set_ANN_daily_load_parameters()\n")
        print("2. Load data from the source file and prepare it for the models using:")
        print("   load_data_in_models()\n")
        print("3. Train the models in one of two ways:")
        print("   a. training_all_models() to train all models simultaneously.")
        print("   b. Train each model individually by:")
        print("      - training_AE()")
        print("      - training_ANN_HV()")
        print("      - training_ANN_DL()\n")
        print("4. To access daily profiles not used in the training, use:")
        print("   get_test_dataframe()\n")
        print("5. Predict the daily profile by calling:")
        print("   predict_profile(day_of_year, type_of_day, decimal_randomness)\n")
        print("Follow these steps to optimize the use of the class and achieve effective results.")
