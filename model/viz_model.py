from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mat_interp.utils.mce import calculate_entropy


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
) -> None:
    """_summary_

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    temperature_column : str
        _description_
    field_column : str
        _description_
    magnetization_column : str
        _description_
    """
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
        ax[1].plot(df_plot.t, df_plot.ds, label=rf"$mu_{0}\Delta H$ = {field} T")

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
    model: Callable, start: float, end: float, step: float
) -> pd.DataFrame:
    """...

    Parameters
    ----------
    model : Callable
        _description_
    start : float
        _description_
    end : float
        _description_
    step : float
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    fields = np.arange(start, end + step, step) * 1e4  # Go to Oe
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
