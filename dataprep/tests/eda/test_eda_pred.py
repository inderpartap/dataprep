"""
    module for testing plot_prediction(df, x, y) function.
"""
import random
import numpy as np
import pandas as pd

from ...eda.eda_plot_pred import plot_prediction


def test_plot_prediction_df() -> None:
    """
    :return:
    """
    df_data = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data['b'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['c'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['d'] = df_data['a'] + np.random.normal(0, 10, 100)
    intermediate = plot_prediction(
        pd_data_frame=df_data,
        target='b',
        return_intermediate=True
    )
    corr_value = df_data.corr()
    for _, tu_pred in enumerate(intermediate.result['pred_score']):
        assert np.isclose(corr_value['b'][tu_pred[0]], tu_pred[1])


def test_plot_prediction_df_x() -> None:
    """
    :return:
    """
    df_data = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data['b'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['c'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['d'] = df_data['a'] + np.random.normal(0, 10, 100)
    _ = plot_prediction(
        pd_data_frame=df_data,
        target='b',
        x_name='c',
        return_intermediate=True
    )

    chars = ['M', 'F', 'N', 'L']
    df_data = pd.DataFrame({'a': pd.Categorical([random.choice(chars)
                                                 for _ in range(100)])})
    df_data['b'] = np.random.normal(0, 10, 100)
    df_data['c'] = np.random.normal(0, 10, 100)
    df_data['d'] = np.random.normal(0, 10, 100)
    _ = plot_prediction(
        pd_data_frame=df_data,
        target='a',
        x_name='b',
        return_intermediate=True
    )


def test_plot_prediction_df_x_y() -> None:
    """
    :return:
    """
    df_data = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data['b'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['c'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['d'] = df_data['a'] + np.random.normal(0, 10, 100)
    _ = plot_prediction(
        pd_data_frame=df_data,
        target='a',
        x_name='b',
        y_name='c',
        return_intermediate=True
    )

    chars = ['M', 'F', 'N', 'L']
    df_data = pd.DataFrame({'a': pd.Categorical([random.choice(chars)
                                                 for _ in range(100)])})
    df_data['b'] = np.random.normal(0, 10, 100)
    df_data['c'] = np.random.normal(0, 10, 100)
    df_data['d'] = np.random.normal(0, 10, 100)
    _ = plot_prediction(
        pd_data_frame=df_data,
        target='a',
        x_name='b',
        y_name='c',
        return_intermediate=True
    )
