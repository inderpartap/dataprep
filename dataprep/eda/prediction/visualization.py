"""
    This module implements the plot_prediction(df, target) function's
    visualization part
"""
import holoviews as hv
from bokeh.layouts import gridplot
from bokeh.models import HoverTool
from bokeh.models.annotations import Title
from bokeh.plotting import Figure

from ...utils import DataType
from ..common import Intermediate


def _vis_pred_corr(
        intermediate: Intermediate
) -> Figure:
    """
    :param intermediate: An object to encapsulate the
    intermediate results.
    :return: A figure object
    """
    result = intermediate.result
    raw_data = intermediate.raw_data
    hv.extension('bokeh', logo=False)
    tooltips = [
        ("Columns Name", "@Columns"),
        ("Score", "@Score")
    ]
    hover = HoverTool(tooltips=tooltips)
    bars = hv.Bars(result['pred_score'], ['Columns'], 'Score').opts(
        tools=[hover]
    )
    fig = hv.render(bars, backend='bokeh')
    title = Title()
    title.text = 'Prediction Score to {}'.format(raw_data['target'])
    fig.title = title
    fig.toolbar_location = None
    fig.toolbar.active_drag = None
    fig.xaxis.visible = False
    return fig


def _vis_pred_stat(  # pylint: disable=too-many-locals
                     # pylint: disable=too-many-statements
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
        title_max = Title()
        title_max.text = 'Max'
        bars_max_fig.title = title_max
        bars_min = hv.Bars([(i + 1, j) for i, j in
                            enumerate(result['stats_comp'][1][x_name].values)],
                           kdims=[target], vdims=[x_name])
        bars_min_fig = hv.render(bars_min, backend='bokeh')
        title_min = Title()
        title_min.text = 'Min'
        bars_min_fig.title = title_min
        bars_sum = hv.Bars([(i + 1, j) for i, j in
                            enumerate(result['stats_comp'][2][x_name].values)],
                           kdims=[target], vdims=[x_name])
        bars_sum_fig = hv.render(bars_sum, backend='bokeh')
        title_sum = Title()
        title_sum.text = 'Sum'
        bars_sum_fig.title = title_sum
        if len(result['stats_comp']) == 4:
            bars_mean = hv.Bars([(i + 1, j) for i, j in
                                 enumerate(result['stats_comp'][3][x_name].values)],
                                kdims=[target], vdims=[x_name])
            bars_mean_fig = hv.render(bars_mean, backend='bokeh')
            title_mean = Title()
            title_mean.text = 'Mean'
            bars_mean_fig.title = title_mean
            fig = gridplot([[bars_max_fig, bars_min_fig],
                            [bars_sum_fig, bars_mean_fig]])
        else:
            fig = gridplot([[bars_max_fig, bars_min_fig],
                            [bars_sum_fig, None]])
    else:
        bars_max = hv.Bars(result['stats_comp'][0],
                           kdims=[target], vdims=[x_name])
        bars_max_fig = hv.render(bars_max, backend='bokeh')
        title_max = Title()
        title_max.text = 'Max'
        bars_max_fig.title = title_max
        bars_min = hv.Bars(result['stats_comp'][1],
                           kdims=[target], vdims=[x_name])
        bars_min_fig = hv.render(bars_min, backend='bokeh')
        title_min = Title()
        title_min.text = 'Min'
        bars_min_fig.title = title_min
        bars_sum = hv.Bars(result['stats_comp'][2],
                           kdims=[target], vdims=[x_name])
        bars_sum_fig = hv.render(bars_sum, backend='bokeh')
        title_sum = Title()
        title_sum.text = 'Sum'
        bars_sum_fig.title = title_sum
        if len(result['stats_comp']) == 4:
            bars_mean = hv.Bars(result['stats_comp'][3],
                                kdims=[target], vdims=[x_name])
            bars_mean_fig = hv.render(bars_mean, backend='bokeh')
            title_mean = Title()
            title_mean.text = 'Mean'
            bars_mean_fig.title = title_mean
            fig = gridplot([[bars_max_fig, bars_min_fig],
                            [bars_sum_fig, bars_mean_fig]])
        else:
            fig = gridplot([[bars_max_fig, bars_min_fig],
                            [bars_sum_fig, None]])
    return fig


def _vis_pred_relation(  # pylint: disable=too-many-locals
                         # pylint: disable=too-many-statements
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
    tooltips = [
        ('Target', '@Target'),
        ('Group', '@Group'),
        ('Count', '@Count')
    ]
    hover = HoverTool(tooltips=tooltips)
    if target_type == DataType.TYPE_NUM:
        # TODO: Optimize this plot code, it is too complex
        bars_max = hv.Bars([(str(chr(i + 65)),
                             intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][0][
                                 intermediate.raw_data[f'{column_name[j]}_name']
        ].values[i])
            for i, _ in enumerate(keys) for j in range(2)],
            ['Target', 'Group'], ['Count']).opts(tools=[hover])
        bars_max_fig = hv.render(bars_max, backend='bokeh')
        title_max = Title()
        title_max.text = 'Max'
        bars_max_fig.title = title_max
        bars_max_fig.xaxis.visible = False
        bars_min = hv.Bars([(str(chr(i + 65)),
                             intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][1][
                                 intermediate.raw_data[f'{column_name[j]}_name']
        ].values[i])
            for i, _ in enumerate(keys) for j in range(2)],
            ['Target', 'Group'], ['Count']).opts(tools=[hover])
        bars_min_fig = hv.render(bars_min, backend='bokeh')
        title_min = Title()
        title_min.text = 'Min'
        bars_min_fig.title = title_min
        bars_min_fig.xaxis.visible = False
        bars_sum = hv.Bars([(str(chr(i + 65)),
                             intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][2][
                                 intermediate.raw_data[f'{column_name[j]}_name']
        ].values[i])
            for i, _ in enumerate(keys) for j in range(2)],
            ['Target', 'Group'], ['Count']).opts(tools=[hover])
        bars_sum_fig = hv.render(bars_sum, backend='bokeh')
        title_sum = Title()
        title_sum.text = 'Sum'
        bars_sum_fig.title = title_sum
        bars_sum_fig.xaxis.visible = False
        if len(result['stats_comp_x']) == 4:
            bars_mean = hv.Bars([(str(chr(i + 65)),
                                  intermediate.raw_data[f'{column_name[j]}_name'],
                                  result[f'stats_comp_{column_name[j]}'][3][
                                      intermediate.raw_data[f'{column_name[j]}_name']
            ].values[i])
                for i, _ in enumerate(keys) for j in range(2)],
                ['Target', 'Group'], ['Count']).opts(tools=[hover])
            bars_mean_fig = hv.render(bars_mean, backend='bokeh')
            title_mean = Title()
            title_mean.text = 'Mean'
            bars_mean_fig.title = title_mean
            bars_mean_fig.xaxis.visible = False
    else:
        bars_max = hv.Bars([(keys[i], intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][0][
                                 intermediate.raw_data[f'{column_name[j]}_name']
        ].values[i])
            for i, _ in enumerate(keys) for j in range(2)],
            ['Target', 'Group'], ['Count']).opts(tools=[hover])
        bars_max_fig = hv.render(bars_max, backend='bokeh')
        title_max = Title()
        title_max.text = 'Max'
        bars_max_fig.title = title_max
        bars_max_fig.xaxis.visible = False
        bars_min = hv.Bars([(keys[i], intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][1][
                                 intermediate.raw_data[f'{column_name[j]}_name']
        ].values[i])
            for i, _ in enumerate(keys) for j in range(2)],
            ['Target', 'Group'], ['Count']).opts(tools=[hover])
        bars_min_fig = hv.render(bars_min, backend='bokeh')
        title_min = Title()
        title_min.text = 'Max'
        bars_min_fig.title = title_min
        bars_min_fig.xaxis.visible = False
        bars_sum = hv.Bars([(keys[i], intermediate.raw_data[f'{column_name[j]}_name'],
                             result[f'stats_comp_{column_name[j]}'][2][
                                 intermediate.raw_data[f'{column_name[j]}_name']
        ].values[i])
            for i, _ in enumerate(keys) for j in range(2)],
            ['Target', 'Group'], ['Count']).opts(tools=[hover])
        bars_sum_fig = hv.render(bars_sum, backend='bokeh')
        title_sum = Title()
        title_sum.text = 'Sum'
        bars_sum_fig.title = title_sum
        bars_sum_fig.xaxis.visible = False
        if len(result['stats_comp_x']) == 4:
            bars_mean = hv.Bars([(keys[i], intermediate.raw_data[f'{column_name[j]}_name'],
                                  result[f'stats_comp_{column_name[j]}'][3][
                                      intermediate.raw_data[f'{column_name[j]}_name']
            ].values[i])
                for i, _ in enumerate(keys) for j in range(2)],
                ['Target', 'Group'], ['Count']).opts(tools=[hover])
            bars_mean_fig = hv.render(bars_mean, backend='bokeh')
            title_mean = Title()
            title_mean.text = 'Max'
            bars_mean_fig.title = title_mean
            bars_mean_fig.xaxis.visible = False
    if 'scatter_location' in intermediate.result.keys():
        scatter_location = intermediate.result['scatter_location']
        list_key = list(scatter_location.keys())
        scatter = hv.Scatter(scatter_location[list_key[0]], label='class_0')
        for i in range(1, len(list_key)):
            scatter = scatter * hv.Scatter(scatter_location[list_key[i]],
                                           label='class_{}'.format(i))
        scatter_fig = hv.render(scatter, backend='bokeh')
        title_scatter = Title()
        title_scatter.text = 'Scatter'
        scatter_fig.title = title_scatter
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
