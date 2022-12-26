import os
import shutil
from typing import List, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback
from optuna.samplers import TPESampler
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.regularizers import L1, L2
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
    n_layers: int,
    n_nodes: int,
    starting_lr: float,
    X_train: np.array,
    solver: str,
    regularizer: Union[str, None],
    regularization_value: float = 0.0,
) -> Model:
    """
    Returns a dense sequential model for a given input.

    Parameters
    ----------
    n_layers : int
        Number of layers in the model.
    n_nodes : int
        Number of nodes in each layer of the model.
    starting_lr : float
        Initial learning rate for the model.
    X_train : np.array
        Training data for the model.
    solver : str
        Solver to use for training the model. Accepted values are 'SGD', 'NAG', 'RMSprop', and 'Adam'.
    regularizer : Union[str, None]
        Regularization method to use for the model. Accepted values are 'l1', 'l2', and 'dropout'.
        If None, no regularization is used.
    regularization_value : float, optional
        Regularization value to use. Default is 0.0.
        If regularization chosen is "dropout" the value is the probability.

    Returns
    -------
    Model
        A sequential Keras model.
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
    # Regularizers
    regularizers = {
        "l1": L1(regularization_value),
        "l2": L2(regularization_value),
    }
    # Get regularization if is not None. Otherwise we just pass that directly...
    regularization_function = None
    if regularizer in ["l1", "l2"]:
        regularization_function = regularizers[regularizer]

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
        layers.append(
            Dense(
                n_nodes,
                activation=activation,
                name=f"layer{i}",
                kernel_regularizer=regularization_function,
            )
        )
        if regularizer == "dropout":
            layers.append(Dropout(regularization_value))
    # Add output layer. Here we use only one layer as we are predicting a single target
    layers.append(Dense(1, name=f"output-layer"))

    # Build up final model and compile
    model = keras.Sequential(layers)
    model.compile(
        loss=mse, optimizer=opt, metrics=[RSquare(dtype=tf.float32, y_shape=(1,))]
    )

    return model


def optimize_neural_net(
    epochs: int,
    solver: str,
    X_train: np.array,
    y_train: np.array,
    num_trials: int,
    working_dir: str,
    nodes_range: Tuple[int, int] = (50, 200),
    layers_range: Tuple[int, int] = (3, 10),
    regularizer: Union[str, None] = None,
    regularization_value: Union[str, float] = 0.0,
):
    """
    Searches for the optimal architecture and hyperparameters for a neural network using Optuna.

    Parameters
    ----------
    epochs : int
        Number of epochs to train the model.
    solver : str
        Solver to use for training the model. Accepted values are 'SGD', 'NAG', 'RMSprop', and 'Adam'.
    X_train : np.array
        Training data for the model.
    y_train : np.array
        Target data for the model.
    num_trials : int
        Number of trials to run for tllable,
        Range of number of layers to consider for the model. Usage: [min, max]
    working_dir : str
        Directory to save intermediate results during the optimization process.
    regularizer : Union[str, None], optional
        Regularization method to use for the model. Accepted values are 'l1', 'l2', and 'dropout'.
        If None, no regularization is used. Default is None.
    regularization_value : float, optional
        Regularization value to use. Default is 0.0. If using 'dropout' value is the probability.
        Can be passed as 'optimize' and the value will be search within the following ranges: [0.1, 0.01, 0.001, 1e-4, 1e-5]. Only works for "l1" or "l2".

    Returns
    -------
        None
    """

    # Training related paramaeters to be used
    num_epochs = epochs
    solver = solver
    X_train = X_train
    y_train = y_train
    training_data_size = len(y_train)

    # Normalize y_train inputs to make it faster
    y_train_std = (y_train - y_train.mean()) / y_train.std()

    # Optimization related paramaters:
    num_trials = num_trials
    num_nodes_range = nodes_range
    num_layers_range = layers_range

    # Directory. Get desired name and create if necessary
    working_dir = working_dir
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
        num_layers = trial.suggest_int(
            "num_layers", num_layers_range[0], num_layers_range[1]
        )
        num_nodes = trial.suggest_int(
            "num_nodes_per_layer", num_nodes_range[0], num_nodes_range[1]
        )
        starting_lr = trial.suggest_loguniform("starting_lr", 0.0005, 0.005)

        # Check regularization
        if regularizer and regularization_value == "optimize":
            if regularizer in ["l1", "l2"]:
                reg_value = trial.suggest_categorical(
                    "reg_value", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0]
                )
            else:
                reg_value = trial.suggest_categorical(
                    "drop_prob", [0.1, 0.2, 0.3, 0.4, 0.5]
                )
        else:
            reg_value = regularization_value

        # # Create the model
        # print(
        #     f"Creating nn with layers: {num_layers} and nodes: {num_nodes} starting with learning rate {starting_lr} for trial {trial.number}.... regularizaiton value is set to {reg_value}..."
        # )
        model = generate_model(
            n_layers=num_layers,
            n_nodes=num_nodes,
            starting_lr=starting_lr,
            X_train=X_train,
            solver=solver,
            regularizer=regularizer,
            regularization_value=reg_value,
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
