import functools

import numpy as np

from nosekatreport import Aqf

def meth_end_aqf(meth):
    """Decorates a test method to ensure that Aqf.end() is called after the test"""
    @functools.wraps(meth)
    def decorated(*args, **kwargs):
        try:
            meth(*args, **kwargs)
        finally:
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
        'decimal' kwarg to specify how many digits after the deciman point is compared.

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
    err = np.array(result) - np.array(expected)
    max_err_ind = np.argmax(np.abs(err))
    max_err = err[max_err_ind]
    if max_err >= abs_err:
        Aqf.failed('Absolute error larger than {abs_err}, max error at'
        'index {max_err_ind}, error: {max_err} - {description}'.format(**locals()))
    else:
        Aqf.passed(description)
