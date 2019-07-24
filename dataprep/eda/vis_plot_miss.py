"""
    This module implements the plot_missing(df, x, y) function's
    visualization part
"""
import math
import holoviews as hv
import numpy as np
import scipy.stats
from bokeh.layouts import gridplot
from bokeh.models import HoverTool
from bokeh.plotting import Figure
from dataprep.eda.common import Intermediate
from dataprep.utils import get_type, DataType


def _vis_none_count(
        intermediate: Intermediate
) -> Figure:
    """
    :param intermediate: An object to encapsulate the
    intermediate results
    :return: A figure object
    """
    hv.extension('bokeh', logo=False)
    distribution = intermediate.result['distribution']
    row, col = distribution.shape
    columns_name = list(intermediate.raw_data['df'].columns.values)
    data_d = [(columns_name[i], j, distribution[i, j])
              for i in range(row) for j in range(col)]
    tooltips = [
        ('z', '@z'),
    ]
    hover = HoverTool(tooltips=tooltips)
    heatmap = hv.HeatMap(data_d).redim.range(z=(0, 1))
    heatmap.opts(
        tools=[hover],
        colorbar=True,
        width=325,
        title="distribution of missing value"
    )
    heatmap_fig = hv.render(heatmap, backend='bokeh')
    heatmap_fig.toolbar_location = None
    heatmap_fig.toolbar.active_drag = None
    heatmap_fig.xaxis.axis_label = ''
    heatmap_fig.yaxis.axis_label = ''
    heatmap_fig.yaxis.major_tick_line_color = None
    heatmap_fig.yaxis.minor_tick_line_color = None
    heatmap_fig.yaxis.major_label_text_font_size = '0pt'
    count = intermediate.result['count']
    data_b = [(columns_name[i], j) for i, j in enumerate(count)]
    bars = hv.Bars(data_b, hv.Dimension("Column_name"), "Count")
    bars_fig = hv.render(bars, backend='bokeh')
    fig = gridplot([[heatmap_fig, bars_fig]])
    fig.sizing_mode = 'scale_width'
    return fig


def _vis_drop_columns(
        intermediate: Intermediate
) -> Figure:
    """
    :param intermediate: An object to encapsulate the
    intermediate results
    :return: A figure object
    """
    pd_data_frame = intermediate.raw_data['df']
    num_bins = intermediate.raw_data['num_bins']
    df_data_drop = intermediate.result['df_data_drop']
    columns_name = intermediate.result['columns_name']
    hv.extension('bokeh')
    fig_list = np.array([None for _ in range(math.ceil(len(columns_name) / 4) * 4)])
    fig_list = fig_list.reshape(int(len(fig_list) / 4), 4)
    count = 0
    for name in columns_name:
        if get_type(pd_data_frame[name]) == DataType.TYPE_NUM:
            hist_origin = hv.Histogram(
                np.histogram(pd_data_frame[name].values, num_bins)
            ).opts(alpha=0.3)
            hist_drop = hv.Histogram(
                np.histogram(df_data_drop[name].values, num_bins)
            ).opts(alpha=0.3)
            hist_fig = hv.render(
                hist_origin * hist_drop,
                backend='bokeh'
            )
            fig_list[count // 4][count % 4] = hist_fig
        elif get_type(pd_data_frame[name]) == DataType.TYPE_CAT:
            bars_origin = hv.Bars(pd_data_frame[name].value_counts()).opts(
                alpha=0.3
            )
            bars_drop = hv.Bars(df_data_drop[name].value_counts()).opts(
                alpha=0.3
            )
            bars_fig = hv.render(
                bars_origin * bars_drop,
                backend='bokeh'
            )
            fig_list[count // 4][count % 4] = bars_fig
        else:
            raise ValueError("the column's type is error")
    fig = gridplot(list(fig_list))
    return fig


def _vis_drop_y(  # pylint: disable=too-many-locals
        intermediate: Intermediate
) -> Figure:
    """
    :param intermediate: An object to encapsulate the
    intermediate results
    :return: A figure object
    """
    pd_data_frame = intermediate.raw_data['df']
    y_name = intermediate.raw_data['y_name']
    num_bins = intermediate.raw_data['num_bins']
    df_data_drop = intermediate.result['df_data_drop']
    origin_data = pd_data_frame[y_name].values
    drop_data = df_data_drop[y_name].values
    hv.extension('bokeh')
    if get_type(pd_data_frame[y_name]) == DataType.TYPE_NUM:
        hist_data_origin = np.histogram(
            origin_data,
            bins=num_bins
        )
        hist_data_drop = np.histogram(
            drop_data,
            bins=num_bins
        )
        hist_dist_origin = scipy.stats.rv_histogram(hist_data_origin)
        hist_dist_drop = scipy.stats.rv_histogram(hist_data_drop)
        sample_x = np.linspace(
            np.min(origin_data),
            np.max(origin_data),
            100
        )
        pdf_origin = hv.Curve((sample_x, hist_dist_origin.pdf(sample_x)),
                              label='origin PDF')
        cdf_origin = hv.Curve((sample_x, hist_dist_origin.cdf(sample_x)),
                              label='origin CDF')
        pdf_drop = hv.Curve((sample_x, hist_dist_drop.pdf(sample_x)),
                            label='drop PDF')
        cdf_drop = hv.Curve((sample_x, hist_dist_drop.cdf(sample_x)),
                            label='drop CDF')
        hist_origin = hv.Histogram(hist_data_origin).opts(
            alpha=0.3
        )
        hist_drop = hv.Histogram(hist_data_drop).opts(
            alpha=0.3
        )
        fig_hist = hv.render(
            hist_origin * hist_drop,
            backend='bokeh'
        )
        fig_curve = hv.render(
            pdf_drop * pdf_origin * cdf_drop * cdf_origin,
            backend='bokeh'
        )
        box_origin = hv.BoxWhisker(origin_data)
        box_drop = hv.BoxWhisker(drop_data)
        fig_box = hv.render(
            box_origin + box_drop,
            backend='bokeh'
        )
        fig = gridplot([[fig_hist, fig_box, fig_curve]])
    elif get_type(pd_data_frame[y_name]) == DataType.TYPE_CAT:
        bars_origin = hv.Bars(pd_data_frame[y_name].value_counts()).opts(
            alpha=0.3
        )
        bars_drop = hv.Bars(df_data_drop[y_name].value_counts()).opts(
            alpha=0.3
        )
        fig_bars = hv.render(
            bars_origin * bars_drop,
            backend='bokeh'
        )
        fig = gridplot([[fig_bars]])
    else:
        raise ValueError("the column's type is error")
    return fig
