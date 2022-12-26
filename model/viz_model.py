from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mat_interp.utils.mce import calculate_entropy, interpolation_temperature
from tensorflow.keras import Sequential


def plot_entropy_curve(
    data: pd.DataFrame,
    temperature_column: str,
    field_column: str,
    magnetization_column: str,
    sample_mass: float = 0.61 * 1e-3,
    unstandarize: bool = False,
    original_mean: float = 0.04164898552871286,
    original_std: float = 0.029355338257634762,
    n_cols: int = 3,
    bbox_mag=(0.8, -0.15),
    bbox_ds=(0.8, -0.15),
    return_entropy: bool = False,
    t_ranges: List[int] = interpolation_temperature,
) -> None:
    """Plots the magnetization-temperature curve and the entropy curve.

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe containing the temperature, field, and magnetization values.
    temperature_column : str
        The name of the column in the dataframe with the temperature values.
    field_column : str
        The name of the column in the dataframe with the field values.
    magnetization_column : str
        The name of the column in the dataframe with the magnetization values.
    sample_mass : float, optional
        The mass of the sample in grams, by default 0.61 * 1e-3
    unstandarize : bool, optional
        If True, the magnetization values will be unstandarized using the mean and std of the original dataset."""
    # Set up color settings and plot aethestics
    sns.set()
    color_pallete = "husl"

    # If it is necessary to unstandarize the magnetization
    if unstandarize:
        magnetization = data.loc[:, magnetization_column]
        magnetization = magnetization * original_std + original_mean
        data[magnetization_column] = magnetization
        # Now lets set up the units correctly
    df_to_calculate = data.copy()
    df_to_calculate[magnetization_column] /= sample_mass
    df_to_calculate[field_column] /= 1e4  # Pass it to tesla instead of Oe.

    # Calculate the entropy
    entropy_dataframe = calculate_entropy(
        df=df_to_calculate,
        t_column=temperature_column,
        h_column=field_column,
        m_column=magnetization_column,
        t_ranges=t_ranges,
    )
    entropy_dataframe["ds"] = entropy_dataframe["ds"].apply(np.abs)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot M-T data first
    fields_mt = df_to_calculate.loc[:, field_column].unique()
    colors_mt = sns.color_palette(palette="husl", n_colors=len(fields_mt))

    # Set up plot
    for field, color in zip(fields_mt, colors_mt):
        df_plot = df_to_calculate.query(f"{field_column} == @field")
        ax[0].plot(
            df_plot[temperature_column],
            df_plot[magnetization_column],
            "o-",
            mec="black",
            color=color,
            label=f"{field} T",
        )

    # Plot up dS for 1 3 and 5 T.
    fields_ds = [1, 3, 5]
    for field in fields_ds:
        df_plot = entropy_dataframe.query("dh == @field")
        ax[1].plot(
            df_plot.t,
            df_plot.ds,
            "o-",
            mec="black",
            label=rf"$\mu_{0}\Delta H$ = {field} T",
        )

    # Labels and what not
    ax[0].set_title(f"Magnetization Data")
    ax[0].legend(ncol=n_cols, bbox_to_anchor=bbox_mag, title="Applied Field")
    ax[0].set_xlabel("T (K)")
    ax[0].set_ylabel("M (emu/g)")

    # Labels and what not
    ax[1].set_title(f"Entropy Change")
    ax[1].legend(ncol=1, bbox_to_anchor=bbox_ds, title="Field Change")
    ax[1].set_xlabel("T (K)")
    ax[1].set_ylabel(r"$\Delta$S$_{M}$ [J/(kg * K)]")

    if return_entropy:
        return entropy_dataframe


def generate_data(
    model: Sequential,
    x1_interpolation_range: Tuple[float, float] = (10, 60),
    x1_interpolation_step: float = 0.5,
    x1_values: List[int] = interpolation_temperature,
    x2_interpolation_range: Tuple[float, float] = (2000, 50000),
    x2_interpolation_step: float = 2000,
    x2_values: List[float] = None,
    y_train: np.array = None,
    x1_name: str = "t",
    x2_name: str = "h",
    y_name: str = "m",
) -> pd.DataFrame:
    """Generates a dataframe with target values based on the input features X1 and X2 previously trained

    Parameters
    ----------
    model : Sequential
        A Sequential keras Model that takes two features as input and returns a target value.

    x1_interpolation_range : Tuple[float, float], optional
        A tuple specifying the range of values to use for the first feature. The range is inclusive, meaning that the values will include both the start and end values. The default range is (10, 60).

    x1_interpolation_step : float, optional
        The step size for the first feature values. The default step size is 0.5.

    x1_values : List[int], optional
        A list of values to use for the first feature. If provided, the `x1_interpolation_range` and `x1_interpolation_step` parameters are ignored. If set to `None`, the first feature values will be generated using the default range and step size. The default values are 'interpolation_temperature' used to recreate the results previously described.

    x2_interpolation_range : Tuple[float, float], optional
        A tuple specifying the range of values to use for the second feature. The range is inclusive, meaning that the values will include both the start and end values. The default range is (2000, 50000).

    x2_interpolation_step : float, optional
        The step size for the second feature values. The default step size is 2000.

    x2_values : List[float], optional
        A list of values to use for the second feature. If provided, the `x2_interpolation_range` and `x2_interpolation_step` parameters are ignored. If set to `None`, the second feature values will be generated using the default range and step size.

    y_train : np.array, optional
        A NumPy array with the training data for the target values. If provided, the target values generated by the model will be scaled to match the mean and standard deviation of the training data.

    x1_name : str, optional
        The name to use for the first feature column in the output dataframe. The default name is "t".

    x2_name : str, optional
        The name to use for the second feature column in the output dataframe. The default name is "h".

    y_name : str, optional
        The name to use for the target column in the output dataframe. The default name is "m".

    Returns
    -------
    pd.DataFrame
        A dataframe with the first and second feature and the predicted value for the target output."""

    if x1_values is None:
        x1_values = np.arange(
            x1_interpolation_range[0],
            x1_interpolation_range[1] + x1_interpolation_step,
            x1_interpolation_step,
        )
    if x2_values is None:
        x2_values = np.arange(
            x2_interpolation_range[0],
            x2_interpolation_range[1] + x2_interpolation_step,
            x2_interpolation_step,
        )

    dfs = []
    for value in x2_values:
        df_ = pd.DataFrame(data=x1_values, columns=[x1_name])
        df_[x2_name] = value
        dfs.append(df_)
    df_final = pd.concat(dfs)
    df_final[y_name] = model.predict(df_final.loc[:, [x1_name, x2_name]])

    if y_train is not None:
        df_final[y_name] = (df_final[y_name] * y_train.std()) + y_train.mean()

    return df_final
