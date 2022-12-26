from typing import List

import numpy as np
import pandas as pd
from scipy.integrate import trapz
from scipy.interpolate import interp1d

# Interpolation temperatures to be used that matches the measurement of ErCo2.
interpolation_temperature = [
    10.00379214,
    10.68471143,
    11.17631571,
    11.68181286,
    12.16917,
    12.69589571,
    13.19775857,
    13.69482714,
    14.20138857,
    14.69220714,
    15.20449714,
    15.68586143,
    16.19239714,
    16.69597714,
    17.19411857,
    17.69024571,
    18.19750143,
    18.69039143,
    19.19919429,
    19.69392857,
    20.19585714,
    20.69653571,
    21.19798429,
    21.69498143,
    22.19800143,
    22.69294857,
    23.19522143,
    23.68773,
    24.19395,
    24.69549,
    25.20007286,
    25.69511286,
    26.20086714,
    26.69798714,
    27.19198571,
    27.70147286,
    28.20193571,
    28.69352571,
    29.19800714,
    29.69744,
    30.19947429,
    30.83235143,
    31.22704857,
    31.69384857,
    32.19356857,
    32.70343857,
    33.19396,
    33.69481714,
    34.19674571,
    34.69362,
    35.19156429,
    35.70390143,
    36.19203143,
    36.70296857,
    37.20304571,
    37.69656429,
    38.19841714,
    38.68739714,
    39.19619571,
    39.70148714,
    40.19109429,
    40.69798857,
    41.18712143,
    41.68401286,
    42.19074714,
    42.69410286,
    43.20054571,
    43.69373429,
    44.19309,
    44.69024429,
    45.18720286,
    45.69030429,
    46.19645429,
    46.69055143,
    47.18856571,
    47.69446714,
    48.19618143,
    48.69845429,
    49.19083857,
    49.69529429,
    50.19361143,
    50.81921714,
    51.20691,
    51.69594857,
    52.19798429,
    52.69848571,
    53.18834857,
    53.69135286,
    54.18562714,
    54.68636286,
    55.19359857,
    55.69764714,
    56.20116286,
    56.69747714,
    57.18866857,
    57.70383286,
    58.18851143,
    58.68804857,
    59.24674857,
    59.73129714,
    59.99452143,
]


def calculate_entropy(
    df: pd.DataFrame,
    t_column: str = "T",
    h_column: str = "H",
    m_column: str = "M",
    t_step: float = 0.7,
    t_ranges: List[int] = interpolation_temperature,
    interpolate: bool = True,
) -> pd.DataFrame:
    """
    Calculate the magnetic entropy change from magnetization measurements.

    This function calculates the magnetic entropy change for a given set of magnetization measurements by interpolating and
    integrating the data in the M-T format for different fields. It returns a dataframe containing the values of entropy
    change in function of T for different applied fields changes.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas dataframe containing the measurement data in a grid of T,H,M values. The dataframe should have columns
        named `t_column`, `h_column`, and `m_column`, which correspond to the temperature, field, and magnetization data,
        respectively.
    t_column : str, optional
        The name of the column with the temperature data, by default "T".
    h_column : str, optional
        The name of the column with the field data, by default "H".
    m_column : str, optional
        The name of the column with the magnetization data, by default "M".
    t_step : float, optional
        The temperature step size for interpolation. The default value is 0.7.
    t_ranges : List[int], optional
        A list of temperatures to use for interpolation. If not provided, the function will use the minimum and maximum
        temperatures in the dataframe as the range, and generate a list of temperatures with a step size of `t_step`.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the values of entropy change in function of T for different fields. The dataframe has
        columns "ds", "t", and "dh", which correspond to the entropy change, temperature, and field, respectively.

    """

    # First we gather the unique fields. We are going to have to itnerpolate the data in the M-T range first
    unique_fields = df[h_column].unique()

    # Set up new temperatures for interpolation
    if t_ranges is not None:
        t_range = np.array(t_ranges)
    else:
        min_t = np.round(df[t_column].min(), 1)
        max_t = np.round(df[t_column].max(), 1)
        t_range = np.arange(min_t, max_t + t_step, t_step)

    dfs_by_field = []
    # Now we go through each field and interpolate the M,T curves one by one, and take the derivative of Mi/Ti
    for field in unique_fields:
        df_ = df.query(f"{h_column} == @field")
        # Get x,y and set up the linear interpolating function
        x = df_[t_column]
        y = df_[m_column]
        f = interp1d(x, y, fill_value="extrapolate")
        # Now get new values for (mi,ti)
        mi = f(t_range)
        dmi = np.gradient(mi) / np.gradient(t_range)

        df_interp = pd.DataFrame(data=zip(t_range, dmi), columns=["ti", "dmdti"])
        df_interp["h"] = field
        dfs_by_field.append(df_interp)
    # Now we can integrate and get the entropy.
    df_interp = pd.concat(dfs_by_field)
    entropies = []
    for temp in t_range:
        df_temp = df_interp.query("ti == @temp")
        for field in df_temp.h:
            to_integrate = df_temp[df_temp.h <= field]
            ds = trapz(to_integrate["dmdti"], to_integrate["h"])
            entropies.append((ds, temp, field))

    deltas_df = pd.DataFrame(data=entropies, columns=["ds", "t", "dh"])
    return deltas_df
