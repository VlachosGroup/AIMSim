"""Plotting functions"""
import matplotlib.pyplot as plt
from seaborn import kdeplot


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
        color: str
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

    plt.figure()
    plt.rcParams['svg.fonttype'] = 'none'
    kdeplot(similarity_vector, **kwargs)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    if plot_title is not None:
        plt.title(plot_title, fontsize=plot_title_fontsize)
    plt.show(block=False)
