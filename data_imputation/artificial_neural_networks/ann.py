from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, BinaryCrossentropy # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.models import Model # type: ignore
from  tensorflow.keras.callbacks import History # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from numpy import ndarray

class ANN():

    def __init__(self, n_input_parameters : int, n_output_parameters: int, n_hidden_layers : int, n_neuron_hidden_layers : int , name_ann:str, output_layer_activation_function:str="sigmoid") -> None:

        self.n_input_parameters = n_input_parameters
        self.n_output_parameters = n_output_parameters
        self.n_hidden_layers = n_hidden_layers
        self.n_neuron_hidden_layers = n_neuron_hidden_layers
        self.name_ann = name_ann
        self.output_layer_activation_function = output_layer_activation_function
        self.ann_model = Model()
        self.history = History()
        self._build()

    def _build(self) -> None:
        """
        Builds and compiles the ANN model.
        """
        # input layer
        input_layer = Input(shape=(self.n_input_parameters,), name="input_layer")
        x = input_layer

        # hidden layers construction
        for layer_index in range(self.n_hidden_layers):
            x = Dense(self.n_neuron_hidden_layers, activation='relu', name=f"hidden_layer_{layer_index}")(x)

        # Output layer
        output_layer = Dense(self.n_output_parameters, activation=self.output_layer_activation_function, name="output_layer")(x)

        # model definition
        self.ann_model = Model(inputs=input_layer, outputs=output_layer, name = self.name_ann)

        # model compilation
        self.ann_model.compile(optimizer=Adam(), loss='mse')
        

    def training(self, X_train:ndarray, y_train:ndarray, X_val:ndarray, y_val:ndarray, epochs:int = 50, batch_size:int = 256):
        """
        Trains the ANN model.

        Args:
            X_train (ndarray): Training data inputs.
            y_train (ndarray): Training data outputs.
            X_val (ndarray): Validation data inputs.
            y_val (ndarray): Validation data outputs.
            epochs (int): Number of epochs to train the model (default is 50).
            batch_size (int): Batch size for training (default is 256).
        """
        self.history = self.ann_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
        #callbacks=[early_stopping],
        )

        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]

        print(f"Final training loss: {final_loss}")
        print(f"Final validation loss: {final_val_loss}")
        print("Model trained successfully.")

    def view_model(self):
        """
        Prints a summary of the ANN model and saves a plot of the model architecture.
        """
        self.ann_model.summary()
        plot_model(self.ann_model, to_file='model.png', show_shapes=True, show_layer_names=True)

    def test(self, X_test: ndarray, y_test: ndarray) -> dict:
        """
        Tests class's ANN model using the provided test data and returns evaluation metrics.

        Parameters:
        X_test (np.ndarray): Test data, where each example should have the same shape as the ANN input.
        y_test (np.ndarray): Test data, where each example should have the same shape as the ANN input.

        Returns:
        dict: A dictionary containing evaluation metrics such as Mean Squared Error (MSE),
            Mean Absolute Error (MAE), and Binary Crossentropy (if applicable).
        """
        # Predicting the test set using the autoencoder
        y_pred = self.ann_model.predict(X_test)

        # Calculate Mean Squared Error (MSE)
        mse = MeanSquaredError()
        mse_value = mse(y_test, y_pred).numpy()

        # Calculate Mean Absolute Error (MAE)
        mae = MeanAbsoluteError()
        mae_value = mae(y_test, y_pred).numpy()

        # Calculate Binary Crossentropy Loss (only applicable for outputs in [0,1])
        bce = BinaryCrossentropy()
        bce_value = bce(y_test, y_pred).numpy()

        metrics = {"Mean Squared Error": mse_value, "Mean Absolute Error": mae_value,
                "Binary Crossentropy": bce_value }
        
        return metrics
