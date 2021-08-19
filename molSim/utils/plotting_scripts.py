"""Plotting functions"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from seaborn import kdeplot, heatmap

from molSim.exceptions import InvalidConfigurationError


def plot_density(similarity_vector, **kwargs):
    """Plot the similarity density.

    Args:
        similarity_vector (list or numpy ndarray): Vector of similarity scores
            to be plotted.

    kwargs: dict
        Keyword arguments to modify plot. Some common ones:
        xlabel: str
            Label of the x-axis. Default is "Samples"
        ylabel: str
            Label of the y-axis. Default is "Similarity Density"
        xlabel_fontsize: int
            Fontsize of the x-axis label. Default is 20.
        ylabel_fontsize: int
            Fontsize of the y-axis label. Default is 20.
        plot_title: str
            Plot title. Default is None.
        plot_title_fontsize: int
            Fontsize of the title. Default is 24.
        plot_color: str
            Color of the plot.
        shade: bool
            To shade the plot or not.
    """
    plot_title = kwargs.pop("plot_title", None)
    xlabel = kwargs.pop("xlabel", "Samples")
    ylabel = kwargs.pop("ylabel", "Similarity Density")
    plot_title_fontsize = kwargs.pop("plot_title_fontsize", 24)
    xlabel_fontsize = int(kwargs.pop("xlabel_fontsize", 20))
    ylabel_fontsize = int(kwargs.pop("ylabel_fontsize", 20))
    plot_color = kwargs.pop("plot_color", None)

    plt.figure()
    plt.rcParams["svg.fonttype"] = "none"
    kdeplot(similarity_vector, color=plot_color, **kwargs)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    if plot_title is not None:
        plt.title(plot_title, fontsize=plot_title_fontsize)


def plot_heatmap(input_matrix, **kwargs):
    """Plot a heatmap representing the input matrix.

    Args:
        input_matrix (np.ndarray): Matrix to be plotted.

    kwargs: dict
        Keyword arguments to modify plot. Some common ones:
        xlabel: str
            Label of the x-axis. Default is "Samples"
        ylabel: str
            Label of the y-axis. Default is "Similarity Density"
        xlabel_fontsize: int
            Fontsize of the x-axis label. Default is 20.
        ylabel_fontsize: int
            Fontsize of the y-axis label. Default is 20.
        plot_title: str
            Plot title. Default is None.
        plot_title_fontsize: int
            Fontsize of the title. Default is 24.
        color: str
            Color of the plot.
        shade: bool
            To shade the plot or not.

    """
    parameters = {
        "xticklabels": False,
        "yticklabels": False,
        "cmap": "autumn",
        "mask_upper": False,
        "annotate": False,
    }
    parameters.update(**kwargs)
    plt.figure()
    plt.rcParams["svg.fonttype"] = "none"
    mask = None
    if parameters["mask_upper"] is True:
        mask = np.triu(np.ones(shape=input_matrix.shape), k=0)
    heatmap_obj = heatmap(
        input_matrix,
        xticklabels=parameters["xticklabels"],
        yticklabels=parameters["yticklabels"],
        cmap=parameters["cmap"],
        mask=mask,
        annot=parameters["annotate"])


def plot_parity(x, y, **kwargs):
    """Plot parity plot of x vs y.

    Args:
        x (n x 1 np.ndarray): values plotted along x axis
        y (n x 1 np.ndarray): values plotted along y axis

    Returns:
        if kwargs.show_plot set to False, returns pyplot axis.

    """
    plot_params = {
        "alpha": 0.7,
        "s": 100,
        "plot_color": "green",
    }
    if kwargs is not None:
        plot_params.update(kwargs)
    plt.figure()
    plt.rcParams["svg.fonttype"] = "none"
    plt.scatter(
        x=x,
        y=y,
        alpha=plot_params["alpha"],
        s=plot_params["s"],
        c=plot_params["plot_color"],
    )
    max_entry = max(max(x), max(y)) + plot_params.get("offset", 5.0)
    min_entry = min(min(x), min(y)) - plot_params.get("offset", 5.0)
    axes = plt.gca()
    axes.set_xlim([min_entry, max_entry])
    axes.set_ylim([min_entry, max_entry])
    plt.plot(
        [min_entry, max_entry],
        [min_entry, max_entry],
        color=plot_params.get("linecolor", "black"),
    )
    plt.title(
        plot_params.get("title", ""),
        fontsize=plot_params.get("title_fontsize", 24)
    )
    plt.xlabel(
        plot_params.get("xlabel", ""),
        fontsize=plot_params.get("xlabel_fontsize", 20)
    )
    plt.ylabel(
        plot_params.get("ylabel", ""),
        fontsize=plot_params.get("ylabel_fontsize", 20)
    )
    plt.xticks(fontsize=plot_params.get("xticksize", 24))
    plt.yticks(fontsize=plot_params.get("yticksize", 24))
    start, end = axes.get_xlim()
    stepsize = (end - start) / 5
    axes.xaxis.set_ticks(np.arange(start, end, stepsize))
    axes.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
    # set y tick stepsize
    start, end = axes.get_ylim()
    stepsize = (end - start) / 5
    axes.yaxis.set_ticks(np.arange(start, end, stepsize))
    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
    plt.text(
        plot_params.pop('text_x', 0.05), plot_params.pop('text_y', 0.9),
        plot_params.pop('text', None), transform=axes.transAxes,
        fontsize=plot_params.pop('text_fontsize', 16))
    if kwargs.get("show_plot", True):
        pass
    else:
        return axes


def plot_barchart(x, heights, colors, xtick_labels=None, **kwargs):
    """Plot a bar chart

    Args:
        x (list or numpy array): X axis grid.
        heights (list or numpy array): Height of the bars.
        colors (list or str): Plot colors.
        xtick_labels (list, optional): Labels to use for each bar. Default is
            None in which case just the indices of the height are used.
    """
    plot_params = {
        "title": kwargs.pop("title", ""),
        "title_fontsize": kwargs.pop("title_fontsize", 24),
        "xlabel": kwargs.pop("xlabel", ""),
        "xlabel_fontsize": kwargs.pop("xlabel_fontsize", 20),
        "ylabel": kwargs.pop("ylabel", ""),
        "ylabel_fontsize": kwargs.pop("ylabel_fontsize", 20),
        "xticksize": kwargs.pop("xticksize", 24),
        "yticksize": kwargs.pop("yticksize", 24),
    }
    plt.figure()
    plt.tight_layout()
    plt.rcParams["svg.fonttype"] = "none"
    if xtick_labels is None:
        xtick_labels = [_ for _ in range(len(heights))]
    plt.bar(x, height=heights, color=colors, tick_label=xtick_labels, **kwargs)
    plt.title(plot_params["title"], fontsize=plot_params["title_fontsize"])
    plt.xlabel(plot_params["xlabel"], fontsize=plot_params["xlabel_fontsize"])
    plt.ylabel(plot_params["ylabel"], fontsize=plot_params["ylabel_fontsize"])
    plt.xticks(fontsize=plot_params["xticksize"])
    plt.yticks(fontsize=plot_params["yticksize"])


def plot_multiple_barchart(x,
                           heights,
                           colors,
                           legend_labels=None,
                           xtick_labels=None,
                           **kwargs):
    """Plot a bar chart with multiplears per category.

    Args:
        x (list or numpy array): X axis grid.
        heights (list or numpy array): Heights of the sets of bars.
            Size of the array is (n_bars_per_xtick, n_xticks),
        colors (list or str): Plot colors. If list supplied,
            list[0] is used for first series, list[1] is used for
            second series and list[2] is used for third series etc.
        legend_labels (list or numpy array): Array of legend names for
            each bar type. Size is (n_bars_per_xticks). Default is None.
        xtick_labels (list, optional): Labels to use for each bar. Default is
            None in which case just the indices of the heights are used.

    Raises:
        InvalidConfigurationError: If number of colors or legend labels
        supplied is less than (or equal to, for legend_labels) n_bars
        (per xtick).
    """
    plot_params = {
        "title": kwargs.pop("title", ""),
        "title_fontsize": kwargs.pop("title_fontsize", 24),
        "xlabel": kwargs.pop("xlabel", ""),
        "xlabel_fontsize": kwargs.pop("xlabel_fontsize", 20),
        "ylabel": kwargs.pop("ylabel", ""),
        "ylabel_fontsize": kwargs.pop("ylabel_fontsize", 20),
        "xticksize": kwargs.pop("xticksize", 24),
        "yticksize": kwargs.pop("yticksize", 24),
    }
    x = np.array(x)
    heights = np.array(heights)
    bar_width = kwargs.pop('bar_width', 0.2)
    n_bars_per_xtick = heights.shape[0]
    if isinstance(colors, str):
        colors = [colors] * n_bars_per_xtick
    if len(colors) < n_bars_per_xtick:
        raise InvalidConfigurationError(f'{len(colors)} colors supplied '
                                        f'insufficient for '
                                        f'{n_bars_per_xtick} bars')
    plt.figure()
    plt.tight_layout()
    plt.rcParams["svg.fonttype"] = "none"
    if xtick_labels is None:
        xtick_labels = x
    bars = []
    for bar_id in range(n_bars_per_xtick):
        bars.append(plt.bar(x + bar_id*bar_width,
                            heights[bar_id],
                            bar_width,
                            color=colors[bar_id],
                            **kwargs))

    plt.title(plot_params["title"], fontsize=plot_params["title_fontsize"])
    plt.xlabel(plot_params["xlabel"], fontsize=plot_params["xlabel_fontsize"])
    plt.ylabel(plot_params["ylabel"], fontsize=plot_params["ylabel_fontsize"])
    plt.xticks(x + bar_width * ((n_bars_per_xtick-1)/2),
               xtick_labels,
               fontsize=plot_params["xticksize"])
    plt.yticks(fontsize=plot_params["yticksize"])
    if legend_labels is not None:
        if len(legend_labels) != n_bars_per_xtick:
            raise InvalidConfigurationError(f'{len(legend_labels)} legend '
                                            f'labels not sufficient for '
                                            f'{n_bars_per_xticks} bars')
        plt.legend(bars, legend_labels)


def plot_scatter(x, y, **kwargs):
    """Plot scatter plot of x vs y.

    Args:
        x(np.ndarray or list): Values plotted along x axis.
        y(np.ndarray or list): Values plotted along y axis.

    Returns:
        if kwargs.show_plot set to False, returns pyplot axis.

    """
    plot_params = {
        "alpha": 0.7,
        "s": 100,
        "plot_color": "green",
    }
    if kwargs is not None:
        plot_params.update(kwargs)
    plt.figure()
    plt.tight_layout()
    plt.rcParams["svg.fonttype"] = "none"
    plt.scatter(
        x=x,
        y=y,
        alpha=plot_params["alpha"],
        s=plot_params["s"],
        c=plot_params["plot_color"],
    )
    max_entry = max(max(x), max(y)) + plot_params.get("offset", 5.0)
    min_entry = min(min(x), min(y)) - plot_params.get("offset", 5.0)
    axes = plt.gca()
    axes.set_xlim([min_entry, max_entry])
    axes.set_ylim([min_entry, max_entry])
    plt.title(
        plot_params.get("title", ""), fontsize=plot_params.get("title_fontsize", 24)
    )
    plt.xlabel(
        plot_params.get("xlabel", ""), fontsize=plot_params.get("xlabel_fontsize", 20)
    )
    plt.ylabel(
        plot_params.get("ylabel", ""), fontsize=plot_params.get("ylabel_fontsize", 20)
    )
    plt.xticks(fontsize=plot_params.get("xticksize", 24))
    plt.yticks(fontsize=plot_params.get("yticksize", 24))
    start, end = axes.get_xlim()
    stepsize = (end - start) / 5
    axes.xaxis.set_ticks(np.arange(start, end, stepsize))
    axes.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
    # set y tick stepsize
    start, end = axes.get_ylim()
    stepsize = (end - start) / 5
    axes.yaxis.set_ticks(np.arange(start, end, stepsize))
    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
    if kwargs.get("show_plot", True):
        pass
    else:
        return axes
