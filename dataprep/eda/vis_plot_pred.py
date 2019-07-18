"""
    This module implements the plot_prediction(df, target) function's
    visualization part
"""
import holoviews as hv
from bokeh.plotting import Figure
from dataprep.eda.common import Intermediate


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
    bars_max = hv.Bars(result['stats_comp'][0],
                       kdims=[target], vdims=[x_name])
    bars_min = hv.Bars(result['stats_comp'][1],
                       kdims=[target], vdims=[x_name])
    bars_sum = hv.Bars(result['stats_comp'][2],
                       kdims=[target], vdims=[x_name])
    if len(result['stats_comp']) == 4:
        bars_mean = hv.Bars(result['stats_comp'][3],
                            kdims=[target], vdims=[x_name])
        bars_mean.toolbar_location = None
        fig = hv.render(bars_max + bars_min + bars_sum + bars_mean,
                        backend='bokeh')
    else:
        fig = hv.render(bars_max + bars_min + bars_sum,
                        backend='bokeh')
    return fig


def _vis_pred_relation(
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
    y_name = intermediate.raw_data['y_name']
    hv.extension('bokeh', logo=False)
    bars_max, bars_min, bars_sum, bars_mean, scatter = \
        None, None, None, None, None
    bars_max = hv.Bars(result['stats_comp_x'][0],
                       kdims=[target], vdims=[x_name]) * \
               hv.Bars(result['stats_comp_y'][0],
                       kdims=[target], vdims=[y_name])
    bars_min = hv.Bars(result['stats_comp_x'][1],
                       kdims=[target], vdims=[x_name]) * \
               hv.Bars(result['stats_comp_y'][1],
                       kdims=[target], vdims=[y_name])
    bars_sum = hv.Bars(result['stats_comp_x'][2],
                       kdims=[target], vdims=[x_name]) * \
               hv.Bars(result['stats_comp_y'][2],
                       kdims=[target], vdims=[y_name])
    if len(result['stats_comp_x']) == 4:
        bars_mean = hv.Bars(result['stats_comp_x'][3],
                            kdims=[target], vdims=[x_name]) * \
                    hv.Bars(result['stats_comp_y'][3],
                            kdims=[target], vdims=[y_name])
    if 'scatter_location' in intermediate.result.keys():
        scatter_location = intermediate.result['scatter_location']
        list_key = list(scatter_location.keys())
        scatter = hv.Scatter(scatter_location[list_key[0]])
        for i in range(1, len(list_key)):
            scatter = scatter * hv.Scatter(scatter_location[list_key[i]])
    if bars_mean is None and scatter is None:
        fig = hv.render(bars_max + bars_min + bars_sum,
                        backend='bokeh')
    elif bars_mean is not None and scatter is None:
        fig = hv.render(bars_max + bars_min + bars_sum + bars_mean,
                        backend='bokeh')
    elif bars_mean is None and scatter is not None:
        fig = hv.render(bars_max + bars_min + bars_sum + scatter,
                        backend='bokeh')
    else:
        layout = bars_max + bars_min + bars_sum + bars_mean + scatter
        fig = hv.render(layout,
                        backend='bokeh')
    return fig
