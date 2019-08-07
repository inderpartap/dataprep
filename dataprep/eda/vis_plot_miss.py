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
from bokeh.models.annotations import Title
from bokeh.plotting import Figure
from dataprep.eda.common import Intermediate
from dataprep.utils import get_type, DataType


def _vis_none_count(  # pylint: disable=too-many-locals
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
        title="Position of Missing Value"
    )
    heatmap_fig = hv.render(heatmap, backend='bokeh')
    heatmap_fig.toolbar_location = None
    heatmap_fig.toolbar.active_drag = None
    heatmap_fig.xaxis.axis_label = 'Column Name'
    heatmap_fig.yaxis.axis_label = 'Position'
    heatmap_fig.yaxis.major_tick_line_color = None
    heatmap_fig.yaxis.minor_tick_line_color = None
    heatmap_fig.yaxis.major_label_text_font_size = '0pt'
    count = intermediate.result['count']
    data_b = [(columns_name[i], j) for i, j in enumerate(count)]
    bars = hv.Bars(data_b, hv.Dimension("Column Name"), "Frequency")
    bars_fig = hv.render(bars, backend='bokeh')
    title = Title()
    title.text = 'Frequency of Missing Value'
    bars_fig.title = title
    fig = gridplot([[heatmap_fig, bars_fig]])
    fig.sizing_mode = 'scale_width'
    return fig


def _vis_drop_columns(  # pylint: disable=too-many-locals
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
    hv.extension('bokeh', logo=False)
    fig_list = np.array([None for _ in range(math.ceil(len(columns_name) / 4) * 4)])
    fig_list = fig_list.reshape(int(len(fig_list) / 4), 4)
    count = 0
    for name in columns_name:
        if get_type(pd_data_frame[name]) == DataType.TYPE_NUM:
            tooltips = [
                ('Frequency', '@Frequency'),
            ]
            hover = HoverTool(tooltips=tooltips)
            hist_origin = hv.Histogram(
                np.histogram(pd_data_frame[name].values, num_bins),
                label='Original Data'
            ).opts(alpha=0.3, tools=[hover])
            hist_drop = hv.Histogram(
                np.histogram(df_data_drop[name].values, num_bins),
                label='Removed Rows'
            ).opts(alpha=0.3, tools=[hover])
            hist_fig = hv.render(
                hist_origin * hist_drop,
                backend='bokeh'
            )
            hist_fig.xaxis.axis_label = 'Column Name'
            hist_fig.yaxis.axis_label = 'Frequency'
            title = Title()
            title.text = 'Frequency of Value'
            hist_fig.title = title
            fig_list[count // 4][count % 4] = hist_fig
            count = count + 1
        elif get_type(pd_data_frame[name]) == DataType.TYPE_CAT:
            tooltips = [
                ('Frequency', '@c'),
            ]
            hover = HoverTool(tooltips=tooltips)
            bars_origin = hv.Bars(
                pd_data_frame[name].value_counts(),
                label='Original Data'
            ).opts(alpha=0.3, tools=[hover])
            bars_drop = hv.Bars(
                df_data_drop[name].value_counts(),
                label='Removed Rows'
            ).opts(alpha=0.3, tools=[hover])
            bars_fig = hv.render(
                bars_origin * bars_drop,
                backend='bokeh'
            )
            bars_fig.xaxis.axis_label = 'Column Name'
            bars_fig.yaxis.axis_label = 'Frequency'
            title = Title()
            title.text = 'Frequency of Value'
            bars_fig.title = title
            fig_list[count // 4][count % 4] = bars_fig
            count = count + 1
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
    hv.extension('bokeh', logo=False)
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
        pdf_origin = hv.Curve(
            (sample_x, hist_dist_origin.pdf(sample_x)),
            label='Origin PDF'
        )
        cdf_origin = hv.Curve(
            (sample_x, hist_dist_origin.cdf(sample_x)),
            label='Origin CDF'
        )
        pdf_drop = hv.Curve(
            (sample_x, hist_dist_drop.pdf(sample_x)),
            label='Removed PDF'
        )
        cdf_drop = hv.Curve(
            (sample_x, hist_dist_drop.cdf(sample_x)),
            label='Removed CDF'
        )
        tooltips = [
            ('Frequency', '@Frequency'),
        ]
        hover = HoverTool(tooltips=tooltips)
        hist_origin = hv.Histogram(
            hist_data_origin
        ).opts(alpha=0.3, tools=[hover])
        hist_drop = hv.Histogram(
            hist_data_drop
        ).opts(alpha=0.3, tools=[hover])
        fig_hist = hv.render(
            hist_origin * hist_drop,
            backend='bokeh'
        )
        fig_pdf = hv.render(
            pdf_origin * pdf_drop,
            backend='bokeh'
        )
        fig_cdf = hv.render(
            cdf_origin * cdf_drop,
            backend='bokeh'
        )
        group_origin = ['Origin' for _, _ in enumerate(origin_data)]
        group_drop = ['Removed' for _, _ in enumerate(drop_data)]
        group_origin.extend(group_drop)
        tooltips = [
            ('Group', '@Group'),
            ('Value', '@Value')
        ]
        hover = HoverTool(tooltips=tooltips)
        box_mixed = hv.BoxWhisker(
            (group_origin, np.append(origin_data, drop_data)),
            ['Group'], 'Value'
        ).opts(tools=[hover])
        fig_box = hv.render(
            box_mixed,
            backend='bokeh'
        )
        fig = gridplot([[fig_hist, fig_box, fig_pdf, fig_cdf]])
    elif get_type(pd_data_frame[y_name]) == DataType.TYPE_CAT:
        tooltips = [
            ('Frequency', '@c'),
        ]
        hover = HoverTool(tooltips=tooltips)
        bars_origin = hv.Bars(pd_data_frame[y_name].value_counts()).opts(
            alpha=0.3,
            tools=[hover]
        )
        bars_drop = hv.Bars(df_data_drop[y_name].value_counts()).opts(
            alpha=0.3,
            tools=[hover]
        )
        fig_bars = hv.render(
            bars_origin * bars_drop,
            backend='bokeh'
        )
        fig = gridplot([[fig_bars]])
    else:
        raise ValueError("the column's type is error")
    return fig
