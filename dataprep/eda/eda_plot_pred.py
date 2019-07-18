"""
    This module implements the plot_prediction(df, target) function's
    calculating intermediate part
"""
from typing import Any, Optional, Dict

import dask
import dask.dataframe as dd
import logging
import numpy as np
import pandas as pd
from enum import Enum
from dataprep.eda.common import Intermediate

LOGGER = logging.getLogger(__name__)


class DataType(Enum):
    """
        Enumeration for storing the different types of data possible in a column
    """
    TYPE_NUM = 1
    TYPE_CAT = 2
    TYPE_UNSUP = 3


def get_type(data: dd.Series) -> DataType:
    """ Returns the type of the input data.
        Identified types are according to the DataType Enumeration.

    Parameter
    __________
    The data for which the type needs to be identified.

    Returns
    __________
    str representing the type of the data.
    """

    col_type = DataType.TYPE_UNSUP
    try:
        if pd.api.types.is_bool_dtype(data):
            col_type = DataType.TYPE_CAT
        elif pd.api.types.is_numeric_dtype(data) and dask.compute(
                data.dropna().unique().size) == 2:
            col_type = DataType.TYPE_CAT
        elif pd.api.types.is_numeric_dtype(data):
            col_type = DataType.TYPE_NUM
        else:
            col_type = DataType.TYPE_CAT
    except NotImplementedError as error:  # TO-DO
        LOGGER.info("Type cannot be determined due to : %s", error)

    return col_type


def _drop_non_numerical_columns(
        pd_data_frame: pd.DataFrame
) -> pd.DataFrame:
    """
    :param pd_data_frame: the pandas data_frame for
    which plots are calculated for each column.
    :return: the numerical pandas data_frame for
    which plots are calculated for each column.
    """
    drop_list = []
    for column_name in pd_data_frame.columns.values:
        if get_type(pd_data_frame[column_name]) != DataType.TYPE_NUM:
            drop_list.append(column_name)
    pd_data_frame.drop(columns=drop_list)
    return pd_data_frame


def _calc_corr(
        data_a: np.ndarray,
        data_b: np.ndarray
) -> Any:
    """
    :param data_a: the column of data frame
    :param data_b: the column of data frame
    :return: A float value which indicates the
    correlation of two numpy array
    """
    return np.corrcoef(data_a, data_b)[0, 1]


def _calc_df_min(
        pd_grouped: pd.DataFrame
) -> pd.DataFrame:
    """
    :param pd_grouped: The data frame has been grouped by
    :return: The min value of each category
    """
    return pd_grouped.min()


def _calc_df_max(
        pd_grouped: pd.DataFrame
) -> pd.DataFrame:
    """
    :param pd_grouped: The data frame has been grouped by
    :return: The max value of each category
    """
    return pd_grouped.max()


def _calc_df_mean(
        pd_grouped: pd.DataFrame
) -> pd.DataFrame:
    """
    :param pd_grouped: The data frame has been grouped by
    :return: The mean value of each category
    """
    return pd_grouped.mean()


def _calc_df_sum(
        pd_grouped: pd.DataFrame
) -> pd.DataFrame:
    """
    :param pd_grouped: The data frame has been grouped by
    :return: The sum value of each category
    """
    return pd_grouped.sum()


def _calc_pred_corr(
        pd_data_frame: pd.DataFrame,
        target: str
) -> Any:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param target: The target colume name of the data frame
    :return: An object to encapsulate the
    intermediate results.
    """
    cal_matrix = pd_data_frame.values.T
    columns_name = list(pd_data_frame.columns)
    name_idx = columns_name.index(target)
    corr_list = []
    for i, _ in enumerate(columns_name):
        tmp = dask.delayed(_calc_corr)(
            cal_matrix[i, :], cal_matrix[name_idx, :])
        corr_list.append(tmp)
    corr_comp = dask.compute(*corr_list)
    pred_score = [(columns_name[i], corr_comp[i])
                  for i, _ in enumerate(columns_name)]
    result = {'pred_score': pred_score}
    raw_data = {'df': pd_data_frame}
    intermediate = Intermediate(result=result, raw_data=raw_data)
    return intermediate


def _calc_pred_stat(
        pd_data_frame: pd.DataFrame,
        target: str,
        x_name: str,
        target_type: DataType
) -> Any:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param target: The target colume name of the data frame
    :param x_name: The colume name of the data frame
    :return: An object to encapsulate the
    intermediate results.
    """
    x_type = get_type(pd_data_frame[x_name])
    stats_list = []
    if target_type == DataType.TYPE_CAT:
        target_groupby = pd_data_frame.groupby(target)
    elif target_type == DataType.TYPE_NUM:
        min_value = pd_data_frame[target].min()
        max_value = pd_data_frame[target].max()
        target_groupby = pd_data_frame.groupby(
            pd.cut(
                pd_data_frame[target],
                np.arange(min_value,
                          max_value + 1,
                          (max_value - min_value) / 10)
            )
        )
    else:
        raise ValueError("Target column's type should be "
                         "categorical or numerical")
    stats_list.append(dask.delayed(_calc_df_max)(target_groupby))
    stats_list.append(dask.delayed(_calc_df_min)(target_groupby))
    stats_list.append(dask.delayed(_calc_df_sum)(target_groupby))
    if x_type == DataType.TYPE_NUM:
        stats_list.append(dask.delayed(_calc_df_mean)(target_groupby))
    elif x_type == DataType.TYPE_CAT:
        pass
    else:
        raise ValueError("The x column's type should "
                         "be categorical or numerical")
    stats_comp = dask.compute(*stats_list)
    result = {
        'stats_comp': stats_comp
    }
    raw_data = {
        'df': pd_data_frame,
        'target': target,
        'x_name': x_name,
        'target_type': target_type
    }
    intermediate = Intermediate(
        result=result,
        raw_data=raw_data
    )
    return intermediate


def _calc_scatter(
        intermediate: Intermediate
) -> Any:
    """
    :param intermediate:  An object to encapsulate the
    intermediate results.
    :return: An object to encapsulate the
    intermediate results with scatter location.
    """
    df_sel = intermediate.raw_data['df'][[intermediate.raw_data['target'],
                                          intermediate.raw_data['x_name'],
                                          intermediate.raw_data['y_name']]]
    df_sel_value = df_sel.values
    loc_dict: Dict[Any, Any] = dict()
    for num, key in enumerate(df_sel_value[:, 0]):
        if key not in loc_dict.keys():
            loc_dict[key] = []
            loc_dict[key].append((df_sel_value[num, 1],
                                  df_sel_value[num, 2]))
        else:
            loc_dict[key].append((df_sel_value[num, 1],
                                  df_sel_value[num, 2]))
    intermediate.result['scatter_location'] = loc_dict
    print(loc_dict)
    return intermediate


def _calc_pred_relation(
        pd_data_frame: pd.DataFrame,
        target: str,
        x_name: str,
        y_name: str,
        target_type: DataType
) -> Any:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param target: The target colume name of the data frame
    :param x_name: The colume name of the data frame
    :param y_name: The colume name of the data frame
    :return: An object to encapsulate the
    intermediate results.
    """
    x_type = get_type(pd_data_frame[x_name])
    y_type = get_type(pd_data_frame[y_name])
    if target_type != DataType.TYPE_CAT and \
            target_type != DataType.TYPE_NUM:
        raise ValueError("Target column's type should be "
                         "categorical or numerical")
    else:
        intermediate_x = _calc_pred_stat(
            pd_data_frame=pd_data_frame,
            target=target,
            x_name=x_name,
            target_type=target_type
        )
        intermediate_y = _calc_pred_stat(
            pd_data_frame=pd_data_frame,
            target=target,
            x_name=y_name,
            target_type=target_type
        )
        result = {
            'stats_comp_x': intermediate_x.result['stats_comp'],
            'stats_comp_y': intermediate_y.result['stats_comp']
        }
        raw_data = {
            'df': intermediate_x.raw_data['df'],
            'target': intermediate_x.raw_data['target'],
            'x_name': intermediate_x.raw_data['x_name'],
            'y_name': intermediate_y.raw_data['x_name'],
            'target_type': intermediate_x.raw_data['target_type']
        }
        intermediate = Intermediate(
            result=result,
            raw_data=raw_data
        )
        if x_type == DataType.TYPE_NUM and \
                y_type == DataType.TYPE_NUM:
            intermediate = _calc_scatter(
                intermediate=intermediate
            )
        return intermediate


def plot_prediction(
        pd_data_frame: pd.DataFrame,
        target: str,
        x_name: Optional[str] = None,
        y_name: Optional[str] = None,
        return_intermediate: bool = False
) -> Any:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param target: The target colume name of the data frame
    :param x_name: a valid column name of the data frame
    :param y_name: a valid column name of the data frame
    :param return_intermediate: whether show intermediate results to users
    :return: A dict to encapsulate the
    intermediate results.

    match (x_name, y_name)
        case (Some, Some) =>
        case (Some, None) =>
        case (None, None) => heatmap
        otherwise => error
    """
    target_type = get_type(pd_data_frame[target])
    if target_type != DataType.TYPE_NUM and \
            target_type != DataType.TYPE_CAT:
        raise ValueError("Target column's type should "
                         "be categorical or numerical")
    else:
        if x_name is not None and y_name is not None:
            intermediate = _calc_pred_relation(
                pd_data_frame=pd_data_frame,
                target=target,
                x_name=x_name,
                y_name=y_name,
                target_type=target_type
            )
        elif x_name is not None:
            intermediate = _calc_pred_stat(
                pd_data_frame=pd_data_frame,
                target=target,
                x_name=x_name,
                target_type=target_type
            )
        elif x_name is None and y_name is not None:
            raise ValueError("Please give a value to x_name")
        else:
            if target_type == DataType.TYPE_NUM:
                pd_data_frame = _drop_non_numerical_columns(
                    pd_data_frame=pd_data_frame
                )
                intermediate = _calc_pred_corr(
                    pd_data_frame=pd_data_frame,
                    target=target
                )
            else:
                raise ValueError("Target column's type should be numerical")
    if return_intermediate:
        return intermediate
    else:
        pass
