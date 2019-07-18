"""
    module for testing plot(df, x, y) function.
"""
from typing import Any, Dict

from time import time
import random
import numpy as np
import pandas as pd

from ...eda.eda_plot import plot  # dataprep.tests.eda.test_eda
from ...eda.eda_plot_corr import plot_correlation


def test_corner() -> None:
    """

    :return:
    """
    df_2 = pd.DataFrame(
        {"all_nan": [np.nan for _ in range(10)], "all_one": np.ones(10),
         "all_zeros": np.zeros(10), "random": np.array(
             [0.38538395, 0.13609054, 0.15973238, 0.96192966, 0.03708882,
              0.03633855, 0.25260128, 0.72139843, 0.74553949,
              0.41102021])})

    df_1_expected = {"all_one": {"bar_plot": {1.0: 10}},
                     "all_zeros": {"bar_plot": {0.0: 10}},
                     "random": {"bar_plot": np.array([2, 2, 1, 1, 1, 0, 0, 2, 0, 1],
                                                     dtype=np.int64)}}

    res = plot(df_2, force_cat=["all_one", "all_zeros"])

    assert res["all_one"] == df_1_expected["all_one"]
    assert res["all_zeros"] == df_1_expected["all_zeros"]

    df_2 = pd.DataFrame({
        "empty": [],
        "another_empty": []
    })

    df_2_expected: Dict[str, Any] = {'scatter_plot': {}}

    res = plot(df_2, "empty", "another_empty")
    assert res == df_2_expected


def test_plot_corr_df(  # pylint: disable=too-many-locals
) -> None:
    """
    :return:
    """
    data = np.random.rand(100, 20)
    df_data = pd.DataFrame(data)

    start_p_pd = time()
    res = df_data.corr(method='pearson')
    end_p_pd = time()
    print("pd pearson time: ", str(end_p_pd - start_p_pd) + " s")

    start_p = time()
    _, intermediate = plot_correlation(
        pd_data_frame=df_data,
        method='pearson',
        return_intermediate=True
    )
    end_p = time()
    print("our pearson time: ", str(end_p - start_p) + " s")
    assert np.isclose(res, intermediate.result['corr']).all()

    start_s_pd = time()
    res = df_data.corr(method='spearman')
    end_s_pd = time()
    print("pd spearman time: ", str(end_s_pd - start_s_pd) + " s")

    start_s = time()
    _, intermediate = plot_correlation(
        pd_data_frame=df_data,
        method='spearman',
        return_intermediate=True
    )
    end_s = time()
    print("our spearman time: ", str(end_s - start_s) + " s")
    assert np.isclose(res, intermediate.result['corr']).all()

    start_k_pd = time()
    res = df_data.corr(method='kendall')
    end_k_pd = time()
    print("pd kendall time: ", str(end_k_pd - start_k_pd) + " s")

    start_k = time()
    _, intermediate = plot_correlation(
        pd_data_frame=df_data,
        method='kendall',
        return_intermediate=True
    )
    end_k = time()
    print("our kendall time: ", str(end_k - start_k) + " s")
    assert np.isclose(res, intermediate.result['corr']).all()


def test_plot_corr_df_k() -> None:
    """
    :return:
    """
    data = np.random.rand(100, 20)
    df_data = pd.DataFrame(data)
    k = 5
    res = df_data.corr(method='pearson')
    row, _ = np.shape(res)
    res_re = np.reshape(
        np.triu(res, 1),
        (row * row,)
    )
    idx = np.argsort(res_re)
    mask = np.zeros(
        shape=(row * row,)
    )
    for i in range(k):
        if res_re[idx[i]] < 0:
            mask[idx[i]] = 1
        if res_re[idx[-i - 1]] > 0:
            mask[idx[-i - 1]] = 1
    res = np.multiply(res_re, mask)
    res = np.reshape(
        res,
        (row, row)
    )
    res += res.T - np.diag(res.diagonal())
    _, intermediate = plot_correlation(
        pd_data_frame=df_data,
        return_intermediate=True,
        k=k
    )
    assert np.isclose(intermediate.result['corr'], res).all()
    assert np.isclose(intermediate.result['mask'], mask).all()


def test_plot_corr_df_x_k() -> None:
    """
    :return:
    """
    df_data = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data['b'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['c'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['d'] = df_data['a'] + np.random.normal(0, 10, 100)
    x_name = 'b'
    k = 3
    name_list = list(df_data.columns.values)
    idx_name = name_list.index(x_name)
    res_p = df_data.corr(method='pearson').values
    res_p[idx_name][idx_name] = -1
    res_s = df_data.corr(method='spearman').values
    res_s[idx_name][idx_name] = -1
    res_k = df_data.corr(method='kendall').values
    res_k[idx_name][idx_name] = -1
    _, intermediate = plot_correlation(
        pd_data_frame=df_data,
        x_name=x_name,
        return_intermediate=True,
        k=k
    )
    assert np.isclose(sorted(res_p[idx_name], reverse=True)[:k],
                      intermediate.result['pearson']).all()
    assert np.isclose(sorted(res_s[idx_name], reverse=True)[:k],
                      intermediate.result['spearman']).all()
    assert np.isclose(sorted(res_k[idx_name], reverse=True)[:k],
                      intermediate.result['kendall']).all()


def test_plot_corr_df_x_y_k() -> None:
    """
    :return:
    """
    df_data = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data['b'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['c'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['d'] = df_data['a'] + np.random.normal(0, 10, 100)
    x_name = 'b'
    y_name = 'c'
    k = 3
    _ = plot_correlation(
        pd_data_frame=df_data,
        x_name=x_name,
        y_name=y_name,
        return_intermediate=False,
        k=k
    )

    letters = ['a', 'b', 'c']
    df_data_cat = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data_cat['b'] = pd.Categorical([random.choice(letters) for _ in range(100)])
    df_data_cat['c'] = pd.Categorical([random.choice(letters) for _ in range(100)])
    _, intermediate = plot_correlation(
        pd_data_frame=df_data_cat,
        x_name='b',
        y_name='c',
        return_intermediate=True
    )
    assert np.isclose(pd.crosstab(df_data_cat['b'], df_data_cat['c']).values,
                      intermediate.result['cross_table']).all()
