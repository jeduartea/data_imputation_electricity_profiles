# Data Imputation in Electricity Consumption Profiles Through Shape Modeling with Autoencoders

This library leverages the profile synthesizer configuration depicted in Figure 1 to estimate matrices $\mathbf{S}_s$ of hidden variables and $\mathbf{E}_s$ of daily energy consumption using a multi-layer Artificial Neural Network (ANN). The neural network designed to estimate the hidden variables takes the day of the year (DOY) and the type of day (TOD) as inputs, while the neural network for estimating daily load also incorporates these hidden variables alongside the type of day and day of the year as its inputs. Once the hidden variables are estimated, they are fed into the decoder to generate the shape. This shape, along with the daily consumption data, is then scaled to finally produce the synthetic profile as illustrated in Figure 2.

![Profile synthesizer with Artificial Neural Networks](figs/profile_synthesizer.eps)
*Figure 2: Profile synthesizer with Artificial Neural Networks.*

The developed code employs an object-oriented programming paradigm, defining four primary objects: `Data`, `Autoencoder`, `ANN`, and `DataImputation`. The `Data` object is responsible for loading the dataset, performing preprocessing tasks such as cleaning, hot-one encoding, and normalization, and dividing the dataset into training, validation, and test sets. The `Autoencoder` object sets up the autoencoder parameters and manages its training, including the separation into encoder and decoder components. The `ANN` object defines the parameters for the multilayer neural network and oversees its training. The `DataImputation` object, the core component, initializes and utilizes the aforementioned objects to train the neural networks (Autoencoder, ANN daily load, ANN hidden variables) as depicted in Figure 3. Additionally, `DataImputation` also configures the synthetic profile generation model as illustrated in Figure 2 through the `predict_profile` method, which generates a synthetic profile based on the day of the year, type of day, and the degree of randomness.

![Training workflow with Artificial Neural Networks](figs/training_workflow.eps)
*Figure 3: Training workflow with Artificial Neural Networks.*

Javier Duarte
Oscar Duarte
Javier Rosero