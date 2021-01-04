"""Plotting functions"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from seaborn import kdeplot, heatmap


def plot_density(similarity_vector, **kwargs):
    """Plot the similarity density

    Attributes
    ----------
    similarity_vector: list or numpy ndarray
        Vector of similarity scores to be plotted.

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
    plot_title = kwargs.pop('plot_title', None)
    xlabel = kwargs.pop('xlabel', 'Samples')
    ylabel = kwargs.pop('ylabel', 'Similarity Density')
    plot_title_fontsize = kwargs.pop('plot_title_fontsize', 24)
    xlabel_fontsize = int(kwargs.pop('xlabel_fontsize', 20))
    ylabel_fontsize = int(kwargs.pop('ylabel_fontsize', 20))
    plot_color = kwargs.pop('plot_color', None)

    plt.figure()
    plt.rcParams['svg.fonttype'] = 'none'
    kdeplot(similarity_vector, color=plot_color,**kwargs)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    if plot_title is not None:
        plt.title(plot_title, fontsize=plot_title_fontsize)
    plt.show(block=False)


def plot_heatmap(input_matrix, **kwargs):
    """Plot a heatmap representing the input matrix

        Attributes
        ----------
        input_vector: np.ndarray
            Matrix to be plotted.

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
        'xticklabels': False,
        'yticklabels': False,
        'cmap': 'autumn',
        'mask_upper': True,
        'annotate': False
    }
    parameters.update(**kwargs)
    plt.figure()
    plt.rcParams['svg.fonttype'] = 'none'
    mask = None
    if parameters['mask_upper'] is True:
        mask = np.triu(np.ones(shape=input_matrix.shape), k=0)
    heatmap_obj = heatmap(
        input_matrix,
        xticklabels=parameters['xticklabels'],
        yticklabels=parameters['yticklabels'],
        cmap=parameters['cmap'],
        mask=mask,
        annot=parameters['annotate'])
    plt.show(block=False)


def plot_parity(x, y, **kwargs):
    """Plot parity plot of x vs y.

    Parameters
    ----------
    x: n x 1 numpy array: values plotted along x axis
    y: n x 1 numpy array: values plotted along y axis

    Returns
    -------
    if kwargs.show_plot set to False, returns pyplot axis.

    """
    plot_params = {
        'alpha': 0.7,
        's': 100,
        'plot_color': 'green',
    }
    if kwargs is not None:
        plot_params.update(kwargs)
        plt.figure()
    plt.rcParams['svg.fonttype'] = 'none'
    plt.scatter(
        x=x, y=y, alpha=plot_params['alpha'], s=plot_params['s'],
        c=plot_params['plot_color'])
    max_entry = max(max(x), max(y)) + plot_params.get('offset', 5.0)
    min_entry = min(min(x), min(y)) - plot_params.get('offset', 5.0)
    axes = plt.gca()
    axes.set_xlim([min_entry, max_entry])
    axes.set_ylim([min_entry, max_entry])
    plt.plot(
        [min_entry, max_entry],
        [min_entry, max_entry],
        color=plot_params.get('linecolor', 'black'))
    plt.title(
        plot_params.get('title', ''),
        fontsize=plot_params.get('title_fontsize', 24))
    plt.xlabel(
        plot_params.get('xlabel', ''),
        fontsize=plot_params.get('xlabel_fontsize', 20))
    plt.ylabel(
        plot_params.get('ylabel', ''),
        fontsize=plot_params.get('ylabel_fontsize', 20))
    plt.xticks(fontsize=plot_params.get('xticksize', 24))
    plt.yticks(fontsize=plot_params.get('yticksize', 24))
    start, end = axes.get_xlim()
    stepsize = (end - start) / 5
    axes.xaxis.set_ticks(np.arange(start, end, stepsize))
    axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    # set y tick stepsize
    start, end = axes.get_ylim()
    stepsize = (end - start) / 5
    axes.yaxis.set_ticks(np.arange(start, end, stepsize))
    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    plt.show(block=False)
