import numpy as np
import pandas as pd
from scipy.integrate import trapz
from scipy.interpolate import interp1d


def calculate_entropy(
    df: pd.DataFrame,
    t_column: str = "T",
    h_column: str = "H",
    m_column: str = "M",
    t_step: float = 0.7,
) -> pd.DataFrame:
    """Function to calculate the magnetic entropy change from magnetization measurements.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing the measurement data in a grid of T,H,M values
    t_column : str, optional
        The name of the column with the temperature data, by default "T"
    h_column : str, optional
        The name of the column with the field data, by default "H"
    m_column : str, optional
        The name of the column with the magnetization data, by default "M"
    t_step: float, optional
        The temperature step for interpolating.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the values of entropy change in function of T for different fields
    """

    # First we gather the unique fields. We are going to have to itnerpolate the data in the M-T range first
    unique_fields = df[h_column].unique()

    # Set up new temperatures for interpolation
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
        dmi = np.gradient(mi, t_range)

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
