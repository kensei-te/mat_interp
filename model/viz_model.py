from typing import List

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
    t_ranges: List[int] = None,
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


def generate_data_mce(
    model: Sequential,
    start: float,
    end: float,
    step: float,
    t_ranges: List[int] = interpolation_temperature,
) -> pd.DataFrame:
    """Generates a dataframe with magnetization values using a model.

    Parameters
    ----------
    model : Sequential
        A Sequential keras Model that takes a temperature and field value as a feature and returns a magnetization value.
    start : float
        The starting field value in tesla.
    end : float
        The ending field value in tesla.
    step : float
        The step size for the field values in tesla.
    t_ranges : List[int], optional
        A list of temperature values to use in the dataframe, by default uses the interpolation_temperature list for the ErCo2 measured sample. If None, temperature values will be generated from 10 to 60 K with a step size of 0.5 K.

    Returns
    -------
    pd.DataFrame
        A dataframe with the temperature, field, and magnetization values.
    """
    fields = np.arange(start, end + step, step) * 1e4  # Go to Oe
    if t_ranges is not None:
        temperature = t_ranges
    else:
        temperature = np.arange(10, 60 + 0.5, 0.5)

    fields_final = [0.01 * 1e4] + list(np.round(fields, 2))

    dfs = []
    for h in fields_final:
        df_ = pd.DataFrame(data=temperature, columns=["t"])
        df_["h"] = h
        dfs.append(df_)
    df_final = pd.concat(dfs)
    df_final["m"] = model.predict(df_final.loc[:, ["t", "h"]])

    return df_final
