from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, BinaryCrossentropy # type: ignore
from tensorflow.keras.models import Model # type: ignore
from  tensorflow.keras.callbacks import History # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from numpy import ndarray, transpose
import matplotlib.pyplot as plt

class Autoencoder:

    def __init__(self, n_input_parameters: int, n_hidden_layers: int, n_neuron_hidden_layers: int, latent_dim:int, name_autoencoder:str)-> None:

        self.name_autoencoder = name_autoencoder       
        self.autoencoder_model = Model()
        self.encoder = Model()
        self.decoder = Model()
        self.history = History()
        self.n_input_parameters = n_input_parameters
        self.n_hidden_layers = n_hidden_layers
        self.n_neuron_hidden_layers = n_neuron_hidden_layers
        self.latent_dim = latent_dim

        self._build()


    def training(self, Xy_train: ndarray, Xy_val: ndarray, epochs: int = 50, batch_size: int = 256) -> None:
        """
        Trains an autoencoder model using provided training and validation data.

        Args:
            X_train (ndarray): Training data used for learning the autoencoder. This data acts as both the input and the target output.
            X_val (ndarray): Validation data used to evaluate the model at the end of each epoch. This helps in tuning the model without overfitting.
            epochs (int, optional): The number of times to iterate over the entire training dataset. Defaults to 50.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 256.

        Returns:
            None: This function does not return any value but updates the `self.history` attribute with the training history object.

        Description:
            This function trains the autoencoder model defined in `self.autoencoder_model`. It uses the data provided in `X_train` for both input and target output, employing the mean squared error loss function typically used in autoencoders to measure the reconstruction error. The function prints a success message upon completion and updates the training history in `self.history`.
        """
        # Training the model using Keras' fit method
        self.history = self.autoencoder_model.fit(
            Xy_train, Xy_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(Xy_val, Xy_val),
            shuffle=True,
            verbose=1
        )
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]

        print(f"Final training loss: {final_loss}")
        print(f"Final validation loss: {final_val_loss}")

        print("Model trained successfully.")
        self._get_encoder()
        self._get_dencoder()

    def test(self, X_test: ndarray) -> dict:
        """
        Tests class's autoencoder model using the provided test data and returns evaluation metrics.

        Parameters:
        X_test (np.ndarray): Test data, where each example should have the same shape as the autoencoder input.

        Returns:
        dict: A dictionary containing evaluation metrics such as Mean Squared Error (MSE),
            Mean Absolute Error (MAE), and Binary Crossentropy (if applicable).
        """
        # Predicting the test set using the autoencoder
        X_pred = self.autoencoder_model.predict(X_test)

        # Calculate Mean Squared Error (MSE)
        mse = MeanSquaredError()
        mse_value = mse(X_test, X_pred).numpy()

        # Calculate Mean Absolute Error (MAE)
        mae = MeanAbsoluteError()
        mae_value = mae(X_test, X_pred).numpy()

        # Calculate Binary Crossentropy Loss (only applicable for outputs in [0,1])
        bce = BinaryCrossentropy()
        bce_value = bce(X_test, X_pred).numpy()

        metrics = {"Mean Squared Error": mse_value, "Mean Absolute Error": mae_value,
                "Binary Crossentropy": bce_value }

        self.plot_test_values(X_test, X_pred)
        
        # Returning the evaluation metrics as a dictionary
        return metrics
    
    def view_model(self):

        self.autoencoder_model.summary()
        #plot_model(self.autoencoder_model, to_file='model.png', show_shapes=True, show_layer_names=True)

    def _build(self) -> None:
        """
        Constructs the autoencoder model by defining its architecture and compiling it.

        This private method sets up the layers of the autoencoder including the encoder,
        latent, and decoder layers. It then compiles the model with the Adam optimizer and
        mean squared error loss.

        The architecture is defined as follows:
        - An input layer that takes data with shape (self.n_input_parameters,).
        - Multiple encoder layers, each with self.n_neuron_hidden_layers and ReLU activation.
        - A latent layer that compresses the data into a representation of self.latent_dim dimensions.
        - Symmetrical decoder layers that attempt to reconstruct the original input.
        - An output layer with a sigmoid activation function to ensure the output values are between 0 and 1.

        The model's encoder and decoder parts are built symmetrically around the latent layer,
        and it is compiled for a regression-like task where the goal is to minimize the difference
        between the input and the reconstructed output.

        Additionally, the method initializes encoder and decoder models for separate usage
        by calling `_get_encoder()` and `_get_dencoder()` methods respectively, which are assumed
        to extract and store the respective parts of the autoencoder.

        Returns:
            None: This method does not return a value but sets the `self.autoencoder_model` attribute.
        """

        # input layer
        input_layer = Input(shape=(self.n_input_parameters,), name="input_layer")
        x = input_layer

        # Encoder construction
        for layer_index in range(self.n_hidden_layers):
            x = Dense(self.n_neuron_hidden_layers, activation='relu', name=f"encoder_layer_{layer_index}")(x)

        # Latent layer
        x = Dense(self.latent_dim, activation='relu', name=f"latent_layer")(x)

        # Decoder construction
        for layer_index in range(self.n_hidden_layers):
            x = Dense(self.n_neuron_hidden_layers, activation='relu', name=f"decoder_layer_{layer_index}")(x)

        # Output layer
        output_layer = Dense(self.n_input_parameters, activation='sigmoid', name="output_layer")(x)

        # model definition
        self.autoencoder_model = Model(inputs=input_layer, outputs=output_layer, name = self.name_autoencoder)

        # model compilation
        self.autoencoder_model.compile(optimizer=Adam(), loss='mse')

        self._get_encoder()
        self._get_dencoder()

    def _get_encoder(self) -> None:
        # El modelo del encoder termina en la capa "latent_layer"
        encoder_output = self.autoencoder_model.get_layer('latent_layer').output
        self.encoder = Model(inputs=self.autoencoder_model.input, outputs=encoder_output, name='encoder')
        
    def _get_dencoder(self) -> None:
        decoder_input = self.autoencoder_model.get_layer('latent_layer').output
        x = decoder_input
        # cuenta las capas que hallan sido nombradas como decoder
        decoder_layer_count = sum(1 for layer in self.autoencoder_model.layers if layer.name.startswith('decoder_layer'))
        for layer in self.autoencoder_model.layers[-(decoder_layer_count+1):]:  # Asumiendo que incluyes la capa de salida también
            x = layer(x)
        self.decoder = Model(inputs=decoder_input, outputs=x, name='decoder')

    def _plotAE(self, ax, X, title):
        ax.plot(transpose(X))
        ax.set_xlabel("Hour")
        ax.set_ylabel("Load")
        ax.set_ylim(0,1)
        if title != "":
            ax.set_title(title)

    def plot_test_values(self, test_values, predicted_values):
        fig,axs=plt.subplots(1,2,figsize=(8,4))
        self._plotAE(axs[0],test_values,"Real")
        self._plotAE(axs[1],predicted_values,"Model")
        plt.tight_layout()