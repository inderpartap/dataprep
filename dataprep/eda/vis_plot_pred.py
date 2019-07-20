"""
    This module implements the plot_prediction(df, target) function's
    visualization part
"""
import holoviews as hv
from bokeh.layouts import gridplot
from bokeh.plotting import Figure
from dataprep.eda.common import Intermediate
from dataprep.utils import DataType


def _vis_pred_corr(
        intermediate: Intermediate
) -> Figure:
    """
    :param intermediate: An object to encapsulate the
    intermediate results.
    :return: A figure object
    """
    result = intermediate.result
    hv.extension('bokeh', logo=False)
    bars = hv.Bars(result['pred_score'], hv.Dimension('Column name'), 'Score')
    fig = hv.render(bars, backend='bokeh')
    fig.toolbar_location = None
    fig.toolbar.active_drag = None
    return fig


def _vis_pred_stat(
        intermediate: Intermediate
) -> Figure:
    """
    :param intermediate: An object to encapsulate the
    intermediate results
    :return: A figure object
    """
    result = intermediate.result
    target = intermediate.raw_data['target']
    x_name = intermediate.raw_data['x_name']
    hv.extension('bokeh', logo=False)
    if intermediate.raw_data['target_type'] == DataType.TYPE_NUM:
        bars_max = hv.Bars([(i + 1, j) for i, j in
                            enumerate(result['stats_comp'][0][x_name].values)],
                           kdims=[target], vdims=[x_name])
        bars_max_fig = hv.render(bars_max, backend='bokeh')
        bars_min = hv.Bars([(i + 1, j) for i, j in
                            enumerate(result['stats_comp'][1][x_name].values)],
                           kdims=[target], vdims=[x_name])
        bars_min_fig = hv.render(bars_min, backend='bokeh')
        bars_sum = hv.Bars([(i + 1, j) for i, j in
                            enumerate(result['stats_comp'][2][x_name].values)],
                           kdims=[target], vdims=[x_name])
        bars_sum_fig = hv.render(bars_sum, backend='bokeh')
        if len(result['stats_comp']) == 4:
            bars_mean = hv.Bars([(i + 1, j) for i, j in
                                 enumerate(result['stats_comp'][3][x_name].values)],
                                kdims=[target], vdims=[x_name])
            bars_mean_fig = hv.render(bars_mean, backend='bokeh')
            fig = gridplot([[bars_max_fig, bars_min_fig],
                            [bars_sum_fig, bars_mean_fig]])
        else:
            fig = gridplot([[bars_max_fig, bars_min_fig],
                            [bars_sum_fig, None]])
    else:
        bars_max = hv.Bars(result['stats_comp'][0],
                           kdims=[target], vdims=[x_name])
        bars_max_fig = hv.render(bars_max, backend='bokeh')
        bars_min = hv.Bars(result['stats_comp'][1],
                           kdims=[target], vdims=[x_name])
        bars_min_fig = hv.render(bars_min, backend='bokeh')
        bars_sum = hv.Bars(result['stats_comp'][2],
                           kdims=[target], vdims=[x_name])
        bars_sum_fig = hv.render(bars_sum, backend='bokeh')
        if len(result['stats_comp']) == 4:
            bars_mean = hv.Bars(result['stats_comp'][3],
                                kdims=[target], vdims=[x_name])
            bars_mean_fig = hv.render(bars_mean, backend='bokeh')
            fig = gridplot([[bars_max_fig, bars_min_fig],
                            [bars_sum_fig, bars_mean_fig]])
        else:
            fig = gridplot([[bars_max_fig, bars_min_fig],
                            [bars_sum_fig, None]])
    return fig


def _vis_pred_relation(  # pylint: disable=too-many-locals
        intermediate: Intermediate
) -> Figure:
    """
    :param intermediate: An object to encapsulate the
    intermediate results
    :return: A figure object
    """
    result = intermediate.result
    target = intermediate.raw_data['target']
    target_type = intermediate.raw_data['target_type']
    hv.extension('bokeh', logo=False)
    bars_max_fig, bars_min_fig, bars_sum_fig, bars_mean_fig, scatter_fig = \
        None, None, None, None, None
    if target_type == DataType.TYPE_NUM:
        keys = result['stats_comp_x'][0][target].values
    else:
        keys = result['stats_comp_x'][0].index.get_level_values(0)
    column_name = ['x', 'y']
    if target_type == DataType.TYPE_NUM:
        # TODO: Optimize this plot code, it is too complex
        bars_max = hv.Bars([(str(chr(i + 65)),
                             intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][0][
                                 intermediate.raw_data[f'{column_name[j]}_name']
                             ].values[i])
                            for i, _ in enumerate(keys) for j in range(2)],
                           ['Target', 'Group'], ['Count'])
        bars_max_fig = hv.render(bars_max, backend='bokeh')
        bars_min = hv.Bars([(str(chr(i + 65)),
                             intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][1][
                                 intermediate.raw_data[f'{column_name[j]}_name']
                             ].values[i])
                            for i, _ in enumerate(keys) for j in range(2)],
                           ['Target', 'Group'], ['Count'])
        bars_min_fig = hv.render(bars_min, backend='bokeh')
        bars_sum = hv.Bars([(str(chr(i + 65)),
                             intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][2][
                                 intermediate.raw_data[f'{column_name[j]}_name']
                             ].values[i])
                            for i, _ in enumerate(keys) for j in range(2)],
                           ['Target', 'Group'], ['Count'])
        bars_sum_fig = hv.render(bars_sum, backend='bokeh')
        if len(result['stats_comp_x']) == 4:
            bars_mean = hv.Bars([(str(chr(i + 65)),
                                  intermediate.raw_data[f'{column_name[j]}_name'],
                                  result[f'stats_comp_{column_name[j]}'][3][
                                      intermediate.raw_data[f'{column_name[j]}_name']
                                  ].values[i])
                                 for i, _ in enumerate(keys) for j in range(2)],
                                ['Target', 'Group'], ['Count'])
            bars_mean_fig = hv.render(bars_mean, backend='bokeh')
    else:
        bars_max = hv.Bars([(keys[i], intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][0][
                                 intermediate.raw_data[f'{column_name[j]}_name']
                             ].values[i])
                            for i, _ in enumerate(keys) for j in range(2)],
                           ['Target', 'Group'], ['Count'])
        bars_max_fig = hv.render(bars_max, backend='bokeh')
        bars_min = hv.Bars([(keys[i], intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][1][
                                 intermediate.raw_data[f'{column_name[j]}_name']
                             ].values[i])
                            for i, _ in enumerate(keys) for j in range(2)],
                           ['Target', 'Group'], ['Count'])
        bars_min_fig = hv.render(bars_min, backend='bokeh')
        bars_sum = hv.Bars([(keys[i], intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][2][
                                 intermediate.raw_data[f'{column_name[j]}_name']
                             ].values[i])
                            for i, _ in enumerate(keys) for j in range(2)],
                           ['Target', 'Group'], ['Count'])
        bars_sum_fig = hv.render(bars_sum, backend='bokeh')
        if len(result['stats_comp_x']) == 4:
            bars_mean = hv.Bars([(keys[i], intermediate.raw_data[f'{column_name[j]}_name'],
                                  result[f'stats_comp_{column_name[j]}'][3][
                                      intermediate.raw_data[f'{column_name[j]}_name']
                                  ].values[i])
                                 for i, _ in enumerate(keys) for j in range(2)],
                                ['Target', 'Group'], ['Count'])
            bars_mean_fig = hv.render(bars_mean, backend='bokeh')
    if 'scatter_location' in intermediate.result.keys():
        scatter_location = intermediate.result['scatter_location']
        list_key = list(scatter_location.keys())
        scatter = hv.Scatter(scatter_location[list_key[0]])
        for i in range(1, len(list_key)):
            scatter = scatter * hv.Scatter(scatter_location[list_key[i]])
        scatter_fig = hv.render(scatter, backend='bokeh')
    if bars_mean_fig is None and scatter_fig is None:
        fig = gridplot([[bars_max_fig, bars_min_fig],
                        [bars_sum_fig, None]])
    elif bars_mean_fig is not None and scatter_fig is None:
        fig = gridplot([[bars_max_fig, bars_min_fig],
                        [bars_sum_fig, bars_mean_fig]])
    elif bars_mean_fig is None and scatter_fig is not None:
        fig = gridplot([[bars_max_fig, bars_min_fig],
                        [bars_sum_fig, scatter_fig]])
    else:
        fig = gridplot([[bars_max_fig, bars_min_fig],
                        [bars_sum_fig, bars_mean_fig],
                        [scatter_fig, None]])
    return fig
