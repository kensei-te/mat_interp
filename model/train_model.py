import os
import shutil
from typing import List, Tuple

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback
from optuna.samplers import TPESampler
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow_addons.metrics import RSquare


def load_train_data(
    datapath: str, feature_columns: Tuple[str, str], target_column: str
) -> Tuple[np.array, np.array]:
    """Load and return the training data.

    This function loads the training data from the specified CSV file and returns it as a tuple of two NumPy arrays,
    containing the feature columns and the target column respectively.

    Args:
        datapath: The path to the CSV file containing the training data.
        feature_columns: A tuple of strings specifying the names of the feature columns in the CSV file.
        target_column: A string specifying the name of the target column in the CSV file.

    Returns:
        A tuple of two NumPy arrays containing the feature columns and the target column of the training data,
        respectively.
    """

    data = pd.read_csv(datapath)
    try:
        X_train = data.loc[:, feature_columns]
        y_train = data.loc[:, target_column]
    except KeyError:
        return "Column name not found into the loaded dataset!"
    return X_train, y_train


def get_batchsize_range(data_length: int) -> List[int]:
    """Returns the batchsize range to consdier based on the size of training data.

    Parameters
    ----------
    data_lenght : int
        Size of training data. Same as len(X_train)

    Returns
    -------
    List[int]
        A list of integers to consider as batchsize for model training
    """
    batch_size_ranges = {
        data_length < 256: [32, 64, 128],
        (data_length >= 256) & (data_length < 1024): [64, 128, 256],
        (data_length >= 1024) & (data_length < 4096): [128, 256, 512],
        data_length >= 4096: [256, 512, 1024],
    }
    return batch_size_ranges[True]


def generate_model(
    n_layers: int, n_nodes: int, starting_lr: float, X_train: np.array, solver: str
) -> Model:
    """Returns a Dense Sequential Model for a given imput

    Parameters
    ----------
    n_layers : int
        Number of layers
    n_nodes : int
        Number of nodes
    starting_lr : float
        Initial learning rate
    solver : str
        Solver to use

    Returns
    -------
    Model
        A sequential keras model
    """
    # Define activation function to use
    activation = "relu"
    # Set up learning rate bondaries conditions for constant decay
    optimizers = {
        "SGD": lambda lr: SGD(learning_rate=lr),
        "NAG": lambda lr: SGD(learning_rate=lr, nesterov=True, momentum=0.1),
        "RMSprop": lambda lr: RMSprop(learning_rate=lr),
        "Adam": lambda lr: Adam(learning_rate=lr),
    }
    lr_values = [starting_lr, starting_lr / 5]
    border = [100]
    learning_rate_function = PiecewiseConstantDecay(border, lr_values)

    opt = optimizers[solver](lr=learning_rate_function)

    # Start to build up model
    normalization_layer = Normalization()
    normalization_layer.adapt(X_train)
    layers = [normalization_layer]

    # Now add dense layers
    for i in range(n_layers):
        layers.append(Dense(n_nodes, activation=activation, name=f"layer{i}"))
    # Add output layer. Here we use only one layer as we are predicting a single target
    layers.append(Dense(1, name=f"output-layer"))

    # Build up final model and compile
    model = keras.Sequential(layers)
    model.compile(
        loss=mse, optimizer=opt, metrics=[RSquare(dtype=tf.float32, y_shape=(1,))]
    )

    return model


def optimize_neural_net(neural_net_settings: dict):
    """
    Function to search for a neural network's architecture and hyperparameters using Optuna.

    Parameters
    ----------
    neural_net_settings: dict
        A dictionary containing the training data and training-related parameters for the neural network. The dictionary should contain the following keys and values:
        - 'epochs': The number of epochs to train the neural network for.
        - 'solver': The solver to use for training the neural network.
        - 'X_train': The training data inputs.
        - 'y_train': The training data labels.
        - 'num_trials': The number of trials to run for optimization.
        - 'working_dir': The directory where the intermediate files generated during optimization will be saved.

    Returns
    -------
        None
    """

    # Training related paramaeters to be used
    num_epochs = neural_net_settings["epochs"]
    solver = neural_net_settings["solver"]
    X_train = neural_net_settings["X_train"]
    y_train = neural_net_settings["y_train"]
    training_data_size = len(y_train)

    # Normalize y_train inputs to make it faster
    y_train_std = (y_train - y_train.mean()) / y_train.std()

    # Optimization related paramaters:
    num_trials = neural_net_settings["num_trials"]

    # Directory. Get desired name and create if necessary
    working_dir = neural_net_settings["working_dir"]
    os.makedirs(working_dir, exist_ok=True)

    def objective(trial: optuna.trial.Trial) -> float:
        """Creates the objective function for optimizing the neural net architheture

        Parameters
        ----------
        trial : optuna.trial.Trial
            A Optuna trial object

        Returns
        -------
        float
            The best R2 score achieved by the model
        """
        # Set up the Neural net archichteture search space
        num_layers = trial.suggest_int("num_layers", 3, 10)
        num_nodes = trial.suggest_int("num_nodes_per_layer", 50, 200)
        starting_lr = trial.suggest_loguniform("starting_lr", 0.0005, 0.005)

        # Create the model
        model = generate_model(
            n_layers=num_layers,
            n_nodes=num_nodes,
            starting_lr=starting_lr,
            X_train=X_train,
            solver=solver,
        )

        # Get the batchsize range and select as a choice
        batch_size_range = get_batchsize_range(training_data_size)
        batch = trial.suggest_categorical("batchsize", batch_size_range)

        # Setup name for rsaving the model weights during training procedure
        trial_number = trial.number
        weight_filename = f"weighttemp_{trial_number}.h5"
        weight_path = os.path.join(working_dir, weight_filename)

        # Set up pruning and early stopping
        pruner_callback = TFKerasPruningCallback(trial, "r_square")
        checkpoint_callback = ModelCheckpoint(
            filepath=weight_path, monitor="loss", save_best_only=True
        )
        earlystopping_callback = EarlyStopping(
            monitor="loss", min_delta=1.0e-10, restore_best_weights=True, patience=20
        )

        # Train the model
        history = model.fit(
            X_train,
            y_train_std,
            batch_size=batch,
            verbose=0,
            epochs=num_epochs,
            callbacks=[pruner_callback, checkpoint_callback, earlystopping_callback],
        )

        # Save history
        history_filename = f"histtrial_{trial_number}.csv"
        history_dataframe = pd.DataFrame(history.history)
        history_dataframe.to_csv(os.path.join(working_dir, history_filename))

        # Now get the best
        r2_maximum = max(history.history["r_square"])

        # Save the model with best weights.
        modelname = f"modeltrial_{trial_number}"
        model.load_weights(weight_path)
        model.save(os.path.join(working_dir, modelname))

        return r2_maximum

    # Now lets create an study optimizer
    study_name = "optuna_NNtemp"
    sampler = TPESampler()
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=50, reduction_factor=2)

    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
    )

    print("Starting optimization process...")
    study.optimize(objective, n_trials=num_trials, n_jobs=-1)
    print("Optimization Finished.")

    # TODO: Have to add code to clean up bad trials and temporary files.
    best_trial_number = study.best_trial.number
    best_trial_name = f"modeltrial_{best_trial_number}"

    # Now lets clean up the folder)
    for file in os.listdir(working_dir):
        # Check if its the best name. Skip it if its, this what we want to keep.
        if file == best_trial_name:
            continue

        # Now lets build up the path
        path = os.path.join(working_dir, file)

        # Check if it is a dir or not
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    print(
        f"Fineshed cleaning up files.. best model kept at {os.path.join(working_dir, best_trial_name)}"
    )
