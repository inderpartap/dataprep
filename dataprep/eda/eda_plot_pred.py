"""
    This module implements the plot_prediction(df, target) function's
    calculating intermediate part
"""
from typing import Any, Dict, Optional, Union, Tuple

import dask
import numpy as np
import pandas as pd
from bokeh.io import show
from bokeh.plotting import Figure
from dataprep.eda.common import Intermediate
from dataprep.eda.vis_plot_pred import _vis_pred_corr, \
    _vis_pred_stat, _vis_pred_relation
from dataprep.utils import DataType, get_type, \
    _drop_non_numerical_columns


def _calc_corr(
        data_a: np.ndarray,
        data_b: np.ndarray
) -> np.float64:
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
) -> Intermediate:
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
            data_a=cal_matrix[i, :],
            data_b=cal_matrix[name_idx, :]
        )
        corr_list.append(tmp)
    corr_comp = dask.compute(*corr_list)
    pred_score = [(columns_name[i], corr_comp[i])
                  for i, _ in enumerate(columns_name)]
    result = {
        'pred_score': pred_score
    }
    raw_data = {
        'df': pd_data_frame,
        'target': target
    }
    intermediate = Intermediate(
        result=result,
        raw_data=raw_data
    )
    return intermediate


def _calc_pred_stat(  # pylint: disable=too-many-locals
        pd_data_frame: pd.DataFrame,
        target: str,
        x_name: str,
        target_type: DataType,
        num_bins: int = 10
) -> Intermediate:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param target: The target colume name of the data frame
    :param x_name: The colume name of the data frame
    :param num_bins: The number of bins when the target column is numerical
    :return: An object to encapsulate the
    intermediate results.
    """
    if get_type(pd_data_frame[x_name]) != DataType.TYPE_NUM:
        raise ValueError("The type of data frame is error")
    stats_list = []
    pd_data_frame_cal = pd_data_frame[[target, x_name]]
    if target_type == DataType.TYPE_CAT:
        target_groupby = pd_data_frame_cal.groupby(target)
    elif target_type == DataType.TYPE_NUM:
        min_value = pd_data_frame_cal[target].min()
        max_value = pd_data_frame_cal[target].max()
        target_groupby = pd_data_frame_cal.groupby(
            pd.cut(
                pd_data_frame_cal[target],
                np.arange(
                    min_value,
                    max_value,
                    (max_value - min_value) / num_bins
                )
            )
        )
    else:
        raise ValueError("Target column's type should be "
                         "categorical or numerical")
    stats_list.append(dask.delayed(_calc_df_max)(target_groupby))
    stats_list.append(dask.delayed(_calc_df_min)(target_groupby))
    stats_list.append(dask.delayed(_calc_df_sum)(target_groupby))
    stats_list.append(dask.delayed(_calc_df_mean)(target_groupby))
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
) -> Intermediate:
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
            loc_dict[key].append(
                (df_sel_value[num, 1],
                 df_sel_value[num, 2])
            )
        else:
            loc_dict[key].append(
                (df_sel_value[num, 1],
                 df_sel_value[num, 2])
            )
    intermediate.result['scatter_location'] = loc_dict
    return intermediate


def _calc_pred_relation(
        pd_data_frame: pd.DataFrame,
        target: str,
        x_name: str,
        y_name: str,
        target_type: DataType
) -> Intermediate:
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
    if x_type != DataType.TYPE_NUM or y_type != DataType.TYPE_NUM:
        raise ValueError("The type of data frame is error")
    if target_type not in (DataType.TYPE_CAT, DataType.TYPE_NUM):
        raise ValueError("Target column's type should be "
                         "categorical or numerical")
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
) -> Union[Figure, Tuple[Figure, Any]]:
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
        case (Some, Some) => Bars + Optional(Scatter)
        case (Some, None) => Bars
        case (None, None) => Heatmap
        otherwise => error
    """
    target_type = get_type(pd_data_frame[target])
    if target_type not in (DataType.TYPE_NUM, DataType.TYPE_CAT):
        raise ValueError("Target column's type should "
                         "be categorical or numerical")
    if x_name is not None and y_name is not None:
        intermediate = _calc_pred_relation(
            pd_data_frame=pd_data_frame,
            target=target,
            x_name=x_name,
            y_name=y_name,
            target_type=target_type
        )
        fig = _vis_pred_relation(
            intermediate=intermediate
        )
    elif x_name is not None:
        intermediate = _calc_pred_stat(
            pd_data_frame=pd_data_frame,
            target=target,
            x_name=x_name,
            target_type=target_type
        )
        fig = _vis_pred_stat(
            intermediate=intermediate
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
            fig = _vis_pred_corr(
                intermediate=intermediate
            )
        else:
            raise ValueError("Target column's type should be numerical")
    show(fig)
    if return_intermediate:
        return fig, intermediate
    return fig
