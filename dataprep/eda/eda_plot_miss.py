"""
    This module implements the plot_missing(df) function's
    calculating intermediate part
"""
from typing import Any, Optional, Union, Tuple

import dask
import numpy as np
import pandas as pd
from bokeh.io import show
from dataprep.eda.common import Intermediate
from dataprep.utils import get_type, DataType


def _calc_none_sum(
        data: np.ndarray,
        length: int
) -> float:
    """
    :param data: A column of data frame
    :param length: The length of array
    :return: The count of None, Nan, Null
    """
    return np.count_nonzero(data) / length


def _calc_none_count(
        pd_data_frame: pd.DataFrame
) -> Intermediate:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :return: An object to encapsulate the
    intermediate results.
    """
    pd_data_frame_value = np.isnan(pd_data_frame.values.T)
    count_none_list = []
    row, col = pd_data_frame_value.shape
    for i in range(row):
        count_none_list.append(
            dask.delayed(_calc_none_sum)(
                pd_data_frame_value[i, :], col
            )
        )
    count_none_comp = dask.compute(*count_none_list)
    result = {
        'distribution': pd_data_frame_value * 1,
        'count': count_none_comp
    }
    raw_data = {
        'df': pd_data_frame
    }
    intermediate = Intermediate(
        result=result,
        raw_data=raw_data
    )
    return intermediate


def _calc_drop_columns(
        pd_data_frame: pd.DataFrame,
        x_name: str,
        num_bins: int = 10
) -> Intermediate:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param x_name: The column whose value missing influence other columns
    :return: An object to encapsulate the
    intermediate results.
    """
    df_data_drop = pd_data_frame.dropna(subset=[x_name])
    columns_name = list(pd_data_frame.columns)
    columns_name.remove(x_name)
    result = {
        'df_data_drop': df_data_drop,
        'columns_name': columns_name
    }
    raw_data = {
        'df': pd_data_frame,
        'x_name': x_name,
        'num_bins': num_bins
    }
    intermediate = Intermediate(
        result=result,
        raw_data=raw_data
    )
    return intermediate


def _calc_drop_y(
        pd_data_frame: pd.DataFrame,
        x_name: str,
        y_name: str,
        num_bins: int = 10
) -> Intermediate:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param x_name:
    :param y_name:
    :return: An object to encapsulate the
    intermediate results.
    """
    df_data_sel = pd_data_frame[[x_name, y_name]]
    df_data_drop = df_data_sel.dropna(subset=[x_name])
    columns_name = list(df_data_drop.columns)
    result = {
        'df_data_drop': df_data_drop,
        'columns_name': columns_name
    }
    raw_data = {
        'df': pd_data_frame,
        'x_name': x_name,
        'y_name': y_name,
        'num_bins': num_bins
    }
    intermediate = Intermediate(
        result=result,
        raw_data=raw_data
    )
    return intermediate


def plot_missing(
        pd_data_frame: pd.DataFrame,
        x_name: Optional[str] = None,
        y_name: Optional[str] = None,
        return_intermediate: bool = False
) -> Any:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param x_name: a valid column name of the data frame
    :param y_name: a valid column name of the data frame
    :param return_intermediate: whether show intermediate results to users
    :return: A dict to encapsulate the
    intermediate results.

    match (x_name, y_name)
        case (Some, Some) => histogram for numerical column,
        bars for categorical column, qq-plot, box-plot, jitter plot,
        CDF, PDF
        case (Some, None) => histogram for numerical column and
        bars for categorical column
        case (None, None) => heatmap
        otherwise => error
    """
    columns_name = list(pd_data_frame.columns)
    for name in columns_name:
        if get_type(pd_data_frame[name]) != DataType.TYPE_NUM and \
                get_type(pd_data_frame[name]) != DataType.TYPE_CAT:
            raise ValueError("the column's data type is error")
    if x_name is not None and y_name is not None:
        intermediate = _calc_drop_y(
            pd_data_frame=pd_data_frame,
            x_name=x_name,
            y_name=y_name
        )
    elif x_name is not None:
        intermediate = _calc_drop_columns(
            pd_data_frame=pd_data_frame,
            x_name=x_name
        )
    elif x_name is None and y_name is not None:
        raise ValueError("Please give a value to x_name")
    else:
        intermediate = _calc_none_count(
            pd_data_frame=pd_data_frame
        )
    if return_intermediate:
        return intermediate
    return None
