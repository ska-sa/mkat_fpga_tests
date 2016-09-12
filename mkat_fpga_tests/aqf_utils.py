import functools
import matplotlib.pyplot as plt
import textwrap

from nosekatreport import Aqf

from mkat_fpga_tests.utils import loggerise
from Tkinter import tkinter

def meth_end_aqf(meth):
    """Decorates a test method to ensure that Aqf.end() is called after the test"""

    @functools.wraps(meth)
    def decorated(*args, **kwargs):
        meth(*args, **kwargs)
        Aqf.end()

    return decorated


def cls_end_aqf(cls):
    """Decorates a test class to ensure that Aqf.end() is called after each test

    Assumes test methods start with test_ or is named runTest
    """
    for attr_name in dir(cls):
        if attr_name.startswith('test_') or attr_name == 'runTest':
            meth = getattr(cls, attr_name)
            if callable(meth):
                setattr(cls, attr_name, meth_end_aqf(meth))
    return cls


def aqf_plot_phase_results(freqs, actual_data, expected_data, plot_filename,
                           plot_title='', plot_units=None, caption='', dump_counts=5,
                           show=False, ):
    """
        Gets actual and expected phase plots.
        return: None
    """
    plt.gca().set_prop_cycle(None)
    if len(actual_data) == dump_counts or len(expected_data) == dump_counts - 1:
        for phases in actual_data:
            plt.plot(freqs, phases)
    else:
        plt.plot(freqs, actual_data[-1], label='{0:.3f} {1}'.format(actual_data[0],
                                                                    plot_units))

    plt.gca().set_prop_cycle(None)
    if len(expected_data) == dump_counts or len(expected_data) == dump_counts - 1:
        if not isinstance(expected_data[0], tuple):
            expected_data = ((expected_data, None),)
        for label_, phases in expected_data:
            fig = plt.plot(
                freqs, phases, '--', label='{0:.3f} {1}'.format(label_, plot_units))[0]
    else:
        fig = plt.plot(freqs, expected_data[-1],
            '--', label='{0:.3f} {1}'.format(expected_data[0], plot_units))[0]

    axes = fig.get_axes()
    ybound = axes.get_ybound()
    yb_diff = abs(ybound[1] - ybound[0])
    new_ybound = [ybound[0] - yb_diff * 1.1, ybound[1] + yb_diff * 1.1]
    # plt.vlines(len(freqs) / 2, *new_ybound, colors='b',
    # linestyles='dotted', label='Center Chan.')
    plt.title('{}'.format(plot_title))
    axes.set_ybound(*new_ybound)
    try:
        plt.grid(True)
    except tkinter.TclError:
        LOGGER.exception('No display on $DISPLAY enviroment variable, check matplotlib backend')
        return False
    else:
        plt.ylabel('Phase [radians]')
        plt.xlabel('Channel number')
        plt.figtext(.1, -.1, '\n'.join(textwrap.wrap(caption)), horizontalalignment='left')
        plt.legend()
        Aqf.matplotlib_fig(plot_filename, caption=caption)
        if show:
            plt.show(block=False)
        plt.clf()


def aqf_plot_channels(channelisation, plot_filename='', plot_title='', log_dynamic_range=90,
                      log_normalise_to=1, normalise=False, caption="", hlines=None, ylimits=None,
                      xlabel=None, show=False):
    """
        Simple magnitude plot of a channelised result
        return: None

        Example
        -------

        aqf_plot_channels(nomalised_magnintude(dump['xeng_raw'][:, 0, :]),
                          'chan_plot_file', 'Channelisation plot')

        If `channelisation` is a tuple it is interpreted as a multi-line plot with
        `channelisation` containing:

        `((plot1_data, legend1), (plot2_data, legend2), ... )`

        If a legend is None it is ignored.

        if `log_dynamic_range` is not None, a log plot will be made with values normalised
        to the peak value of less than -`log_dynamic_range` dB set to -`log_dynamic_range`

        Normalise log dynamic range to `log_normalise_to`. If None, each line is
        normalised to it's own max value, which can be confusing if they don't all have
        the same max...

        If Normalise = True the maximum log value will be subtracted from the loggerised
        data.
    """
    try:
        if not isinstance(channelisation[0], tuple):
            channelisation = ((channelisation, None),)
    except IndexError:
        Aqf.failed('List of channel responses out of range: {}'.format(channelisation))
    has_legend = False
    for plot_data, legend in channelisation:
        kwargs = {}
        if legend:
            has_legend = True
            kwargs['label'] = legend
        if log_dynamic_range is not None:
            plot_data = loggerise(plot_data, log_dynamic_range,
                                  normalise_to=log_normalise_to, normalise=normalise)
            ylabel = 'Channel response [dB]'
        else:
            ylabel = 'Channel response (linear)'

        try:
            plt.grid(True)
        except tkinter.TclError:
            LOGGER.exception('No display on $DISPLAY enviroment variable, check matplotlib backend')
            return False
        plt.plot(plot_data, **kwargs)
        if plot_title:
            plt.title(plot_title)
        plt.ylabel(ylabel)
        if xlabel:
            plt.xlabel(xlabel)
        else:
            plt.xlabel('Channel number')

    if hlines:
        plt.axhline(hlines, linestyle='dotted', color='red', linewidth=1.5)
        msg = ('Channel isolation: {0:.3f} dB'.format(hlines))
        plt.annotate(msg, xy=(len(plot_data) / 2, hlines), xytext=(-20, -30),
                     textcoords='offset points', ha='center', va='bottom',
                     bbox=dict(boxstyle='round, pad=0.2', alpha=0.3),
                     arrowprops=dict(arrowstyle='->', fc='yellow',
                                     connectionstyle='arc3, rad=0.5', color='red'))

    if ylimits:
        plt.ylim(ylimits)

    plt.figtext(.1, -.15, '\n'.join(textwrap.wrap(caption)), horizontalalignment='left')
    if has_legend:
        plt.legend(fontsize=9, fancybox=True,
                   loc='center left', bbox_to_anchor=(1, .8),
                   borderaxespad=0.).set_alpha(0.5)

    Aqf.matplotlib_fig(plot_filename, caption=caption)
    if show:
        fig1 = plt.gcf()  # Get Current Figure
        plt.show(block=False)
        plt.draw()
        fig1.savefig(plot_filename, bbox_inches='tight', dpi=100)
        plt.clf()


def aqf_plot_histogram(data_set, plot_filename='test_plt.png', plot_title=None,
                       caption="", bins=256, ranges=(-1, 1), ylabel='Samples per Bin',
                       xlabel='ADC Sample Bins', show=False):
    """Simple histogram plot of a data set
        return: None
    """
    try:
        plt.grid(True)
    except tkinter.TclError:
        LOGGER.exception('No display on $DISPLAY enviroment variable, check matplotlib backend')
        return False
    else:
        plt.hist(data_set, bins=bins, range=ranges)
        if plot_title:
            plt.title(plot_title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.figtext(.1, -.1, '\n'.join(textwrap.wrap(caption)), horizontalalignment='left')
        Aqf.matplotlib_fig(plot_filename, caption=caption)
        if show:
            plt.show(block=False)
        plt.clf()


def aqf_plot_and_save(freqs, data, df, expected_fc, plot_filename, plt_title,
                      caption="", cutoff=None, show=False):
    fig = plt.plot(freqs, data)[0]
    axes = fig.get_axes()
    ybound = axes.get_ybound()
    yb_diff = abs(ybound[1] - ybound[0])
    # new_ybound = [ybound[0] - yb_diff * 1.1, ybound[1] + yb_diff * 1.1]
    new_ybound = [ybound[0] * 1.1, ybound[1] * 1.1]
    new_ybound = [y if y != 0 else yb_diff * 0.05 for y in new_ybound]
    plt.vlines(expected_fc, *new_ybound, colors='r', label='Channel Fc')
    plt.vlines(expected_fc - df / 2, *new_ybound, label='Channel min/max')
    plt.vlines(expected_fc - 0.8 * df / 2, *new_ybound, label='Channel at +-40%',
               linestyles='--')
    plt.vlines(expected_fc + df / 2, *new_ybound, label='_Channel max')
    plt.vlines(expected_fc + 0.8 * df / 2, *new_ybound, label='_Channel at +40%',
               linestyles='--')
    plt.title(plt_title)
    axes.set_ybound(*new_ybound)
    try:
        plt.grid(True)
    except tkinter.TclError:
        LOGGER.exception('No display on $DISPLAY enviroment variable, check matplotlib backend')
        return False
    else:
        plt.ylabel('dB relative to VACC max')
        # TODO Normalise plot to frequency bins
        plt.xlabel('Frequency (Hz)')
        if cutoff:
            msg = ('Channel isolation: {0:.3f}dB'.format(cutoff))
            plt.axhline(cutoff, color='red', ls='dotted', linewidth=1.5, label=msg)

        plt.figtext(.1, -.1, '\n'.join(textwrap.wrap(caption)), horizontalalignment='left')
        plt.legend(fontsize=9, fancybox=True, loc='center left', bbox_to_anchor=(1, .8),
                   borderaxespad=0.)

        Aqf.matplotlib_fig(plot_filename, caption=caption)
        if show:
            plt.show(block=False)
            plt.clf()
