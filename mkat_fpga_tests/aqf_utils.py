import functools

import numpy as np
import matplotlib.pyplot as plt

from nosekatreport import Aqf

from mkat_fpga_tests.utils import loggerise

def meth_end_aqf(meth):
    """Decorates a test method to ensure that Aqf.end() is called after the test"""
    @functools.wraps(meth)
    def decorated(*args, **kwargs):
        meth(*args, **kwargs)
        Aqf.end()

    return decorated

def cls_end_aqf(cls):
    """Decorates a test class to ensure that Aqf.end() is called after each test

    Assumes test methods start with test_ are is named runTest
    """
    for attr_name in dir(cls):
        if attr_name.startswith('test_') or attr_name == 'runTest':
            meth = getattr(cls, attr_name)
            if callable(meth):
               setattr(cls, attr_name,  meth_end_aqf(meth))
    return cls

def aqf_numpy_almost_equal(result, expected, description, **kwargs):
    """Compares numerical result to an expected value and logs to Aqf.

    Using numpy.testing.assert_almost_equal for the comparison

    Parameters
    ----------
    result: numeric type or array of type
        Actual result to be checked.
    expected: Same as result
        Expected result
    description: String
        Message describing the purpose of the comparison.
    **kwargs : keyword arguments
        Passed on to numpy.testing.assert_almost_equal. You probably want to use the
        `decimal` kwarg to specify how many digits after the decimal point is compared.

    """
    try:
        np.testing.assert_almost_equal(result, expected, **kwargs)
    except AssertionError, e:
        Aqf.failed('{} - {}'.format(str(e), description))
    else:
        Aqf.passed(description)


def aqf_numpy_allclose(result, expected, description, **kwargs):
    """Compares numerical result to an expected value and logs to Aqf.

    Using numpy.testing.assert_allclose for the comparison

    Parameters
    ----------
    result: numeric type or array of type
        Actual result to be checked.
    expected: Same as result
        Expected result
    description: String
        Message describing the purpose of the comparison.
    **kwargs : keyword arguments
        Passed on to numpy.testing.assert_allclose. You probably want to use the
        `rtol` and `atol` kwargs to respectively specify relative- and absolute
        tollerance

    """
    try:
        np.testing.assert_almost_equal(result, expected, **kwargs)
    except AssertionError, e:
        Aqf.failed('{} - {}'.format(str(e), description))
    else:
        Aqf.passed(description)

def aqf_array_abs_error_less(result, expected, description, abs_error=0.1):
    """Compares absolute error in numeric result and logs to Aqf.

    Parameters
    ----------
    result: numeric type or array of type
        Actual result to be checked.
    expected: Same as result
        Expected result
    description: String
        Message describing the purpose of the comparison.
    abs_err: float, optional
        Fail if absolute error is not less than this abs_err for all array
        elements

    """
    err = np.abs(np.array(expected) - np.array(result))
    max_err_ind = np.argmax(err)
    max_err = err[max_err_ind]
    if max_err >= abs_error:
        Aqf.failed('Absolute error larger than {abs_error}, max error at'
        ' index {max_err_ind}, error: {max_err} - {description}'.format(**locals()))
    else:
        Aqf.passed(description)

def aqf_plot_phase_results(freqs, actual_data, expected_data, plot_units,
            plot_filename, plot_title, caption, show=False):
        """
        Gets actual and expected phase plots.
        return: None
        """
        plt.gca().set_prop_cycle(None)
        for phases in actual_data:
            plt.plot(freqs, phases)

        plt.gca().set_prop_cycle(None)
        for label, phases in expected_data:
            fig = plt.plot(
                freqs, phases, '--', label='{} {}'.format(label, plot_units))[0]

        axes = fig.get_axes()
        ybound = axes.get_ybound()
        yb_diff = abs(ybound[1] - ybound[0])
        new_ybound = [ybound[0] - yb_diff*1.1, ybound[1] + yb_diff*1.1]
        plt.vlines(len(freqs)/2, *new_ybound, colors='b',
            linestyles='dotted',label='Center Chan.')
        plt.legend()
        plt.title('{}'.format(plot_title))
        axes.set_ybound(*new_ybound)
        plt.grid(True)
        plt.ylabel('Phase [radians]')
        plt.xlabel('No. of Channels')
        Aqf.matplotlib_fig(plot_filename, caption=caption, close_fig=False)
        if show:
            plt.show()
        plt.close()


def aqf_plot_channels(channelisation, plot_filename, plot_title=None,
                      log_dynamic_range=None, log_normalise_to=None,
                      caption="", show=False):
        """Simple magnitude plot of a channelised result
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

        """
        if not isinstance(channelisation[0], tuple):
            channelisation = ((channelisation, None),)

        has_legend = False
        for plot_data, legend in channelisation:
            kwargs = {}
            if legend:
                has_legend = True
                kwargs['label'] = legend
            if log_dynamic_range is not None:
                plot_data = loggerise(plot_data, log_dynamic_range,
                                      normalise_to=log_normalise_to)
                ylabel = 'Channel response [dB]'
            else:
                ylabel = 'Channel response (linear)'

            plt.plot(plot_data, **kwargs)
            if plot_title:
                plt.title(plot_title)
            plt.ylabel(ylabel)
            plt.xlabel('Channel number')

        axis = plt.gcf().get_axes()[0]
        ybound = axis.get_ybound()
        yb_diff = abs(ybound[1] - ybound[0])
        new_ybound = [ybound[0] - yb_diff*1.1, ybound[1] + yb_diff*1.1]
        #axis.set_ybound(*new_ybound)
        if has_legend:
            plt.legend()

        Aqf.matplotlib_fig(plot_filename, caption=caption, close_fig=False)
        if show:
            plt.show()
        plt.close()
