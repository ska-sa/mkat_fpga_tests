# import threading
import base64
import contextlib
import glob
import pandas as pd
import h5py
import io
import logging
import numpy as np
import operator
import os
import pwd
import Queue
import random
import signal
import subprocess
import time
import warnings
import struct
import socket

from collections import Mapping
from Crypto.Cipher import AES
from inspect import currentframe
from inspect import getframeinfo
# MEMORY LEAKS DEBUGGING
# To use, add @DetectMemLeaks decorator to function
# from memory_profiler import profile as DetectMemLeaks
from nosekatreport import Aqf
from socket import inet_ntoa
from struct import pack

try:
    from collections import ChainMap
except ImportError:
    from chainmap import ChainMap

from casperfpga.utils import threaded_fpga_function
from casperfpga.utils import threaded_fpga_operation
from casperfpga.utils import threaded_create_fpgas_from_hosts
from corr2.data_stream import StreamAddress


# LOGGER = logging.getLogger(__name__)
LOGGER = logging.getLogger('mkat_fpga_tests')

# Max range of the integers coming out of VACC
VACC_FULL_RANGE = float(2**31)

cam_timeout = 60

# Define lambda functions to convert ip to int and back
ip2int = lambda ipstr: struct.unpack('!I', socket.inet_aton(ipstr))[0]
int2ip = lambda n: socket.inet_ntoa(struct.pack('!I', n))

def complexise(input_data):
    """Convert input data shape (X,2) to complex shape (X)
    :param input_data: Xeng_Raw
    """
    return input_data[:, 0] + input_data[:, 1] * 1j


def magnetise(input_data):
    """Convert input data shape (X,2) to complex shape (X) and
       Calculate the absolute value element-wise.
       :param input_data: Xeng_Raw
    """
    id_c = complexise(input_data)
    id_m = np.abs(id_c)
    return id_m


def normalise(input_data):
    return input_data / VACC_FULL_RANGE


def normalised_magnitude(input_data):
    return normalise(magnetise(input_data))


def loggerise(data, dynamic_range=70, normalise=False, normalise_to=None):
    with np.errstate(divide='ignore'):
        log_data = 10 * np.log10(data)
    if normalise_to:
        max_log = normalise_to
    else:
        max_log = np.max(log_data)
    min_log_clip = max_log - dynamic_range
    log_data[log_data < min_log_clip] = min_log_clip
    if normalise:
        log_data = np.asarray(log_data) - np.max(log_data)
    return log_data


def baseline_checker(xeng_raw, check_fn):
    """Apply a test function to correlator data one baseline at a time

    Returns a set of all the baseline indices for which the test matches
    :rtype: dict
    """
    baselines = set()
    for _baseline in range(xeng_raw.shape[1]):
        if check_fn(xeng_raw[:, _baseline, :]):
            baselines.add(_baseline)
    return baselines


def zero_baselines(xeng_raw):
    """Return baseline indices that have all-zero data"""
    return baseline_checker(xeng_raw, lambda bldata: np.all(bldata == 0))


def nonzero_baselines(xeng_raw):
    """Return baseline indices that have some non-zero data"""
    return baseline_checker(xeng_raw, lambda bldata: np.any(bldata != 0))


def all_nonzero_baselines(xeng_raw):
    """Return baseline indices that have all non-zero data"""
    non_zerobls = lambda bldata: np.all(
        np.linalg.norm(
            bldata.astype(np.float64), axis=1) != 0)
    return baseline_checker(xeng_raw, non_zerobls)


def init_dsim_sources(dhost):
    """Select dsim signal output, zero all sources, output scaling to 1

    Also clear noise diode and adc over-range flags
    """
    # Reset flags
    LOGGER.info('Reset digitiser simulator to all Zeros')
    try:
        dhost.registers.flag_setup.write(adc_flag=0, ndiode_flag=0, load_flags='pulse')
    except Exception:
        LOGGER.error('Failed to set dhost flag registers.')
        pass

    try:
        for sin_source in dhost.sine_sources:
            sin_source.set(frequency=0, scale=0)
            assert sin_source.frequency == sin_source.scale == 0
            try:
                if sin_source.name != 'corr':
                    sin_source.set(repeat_n=0)
            except NotImplementedError:
                    LOGGER.exception('Failed to reset repeat on sin_%s' %sin_source.name)
                    import IPython; globals().update(locals()); IPython.embed(header='Python Debugger')
            LOGGER.info('Digitiser simulator cw source %s reset to Zeros' %sin_source.name)
    except Exception:
        LOGGER.error('Failed to reset sine sources on dhost.')
        pass

    try:
        for noise_source in dhost.noise_sources:
            noise_source.set(scale=0)
            assert noise_source.scale == 0
            LOGGER.info('Digitiser simulator awg sources %s reset to Zeros' %noise_source.name)
    except Exception:
        LOGGER.error('Failed to reset noise sources on dhost.')
        pass

    try:
        for output in dhost.outputs:
            output.select_output('signal')
            output.scale_output(1)
            LOGGER.info('Digitiser simulator signal output %s selected.' %output.name)
    except Exception:
        LOGGER.error('Failed to select output dhost.')
        pass


class CorrelatorFrequencyInfo(object):
    """Derive various bits of correlator frequency info using correlator config"""

    def __init__(self, corr_config):
        """Initialise the class

        Parameters
        ==========
        corr_config : dict
            Correlator config dict as in :attr:`corr2.fxcorrelator.FxCorrelator.configd`

        """
        try:
            self.corr_config = corr_config
            self.n_chans = int(corr_config['fengine']['n_chans'])
            assert isinstance(self.n_chans, int)
            # Number of frequency channels
            self.bandwidth = float(corr_config['fengine']['bandwidth'])
            assert isinstance(self.bandwidth, float)
            # Correlator bandwidth
            self.delta_f = self.bandwidth / self.n_chans
            assert isinstance(self.delta_f, float)
            # Spacing between frequency channels
            f_start = 0.  # Center freq of the first bin
            self.chan_freqs = f_start + np.arange(self.n_chans) * self.delta_f
            # Channel centre frequencies
            self.sample_freq = float(corr_config['FxCorrelator']['sample_rate_hz'])
            assert isinstance(self.sample_freq, float)
            self.sample_period = 1 / self.sample_freq
            self.fft_period = self.sample_period * 2 * self.n_chans
            """Time length of a single FFT"""
            self.xeng_accumulation_len = int(corr_config['xengine']['accumulation_len'])
            self.corr_destination, self.corr_rx_port = corr_config['xengine']['output_destinations_base'].split(':')
            self.corr_rx_port = int(self.corr_rx_port)
        except Exception:
            LOGGER.exception('Failed to retrieve various bits of corr freq from corr config')


    def calc_freq_samples(self, chan, samples_per_chan, chans_around=0):
        """Calculate frequency points to sweep over a test channel.

        Parameters
        =========
        chan : int
           Channel number around which to place frequency samples
        samples_per_chan: int
           Number of frequency points per channel
        chans_around: int
           Number of channels to include around the test channel. I.e. value 1 will
           include one extra channel above and one below the test channel.

        Will put frequency sample on channel boundary if 2 or more points per channel are
        requested, and if will place a point in the centre of the chanel if an odd number
        of points are specified.

        """
        try:
            assert samples_per_chan > 0
            assert chans_around > 0
            assert 0 <= chan < self.n_chans
            assert 0 <= chan + chans_around < self.n_chans
            assert 0 <= chan - chans_around < self.n_chans

            start_chan = chan - chans_around
            end_chan = chan + chans_around
            if samples_per_chan == 1:
                return self.chan_freqs[start_chan:end_chan + 1]

            start_freq = self.chan_freqs[start_chan] - self.delta_f / 2
            end_freq = self.chan_freqs[end_chan] + self.delta_f / 2
            sample_spacing = self.delta_f / (samples_per_chan - 1)
            num_samples = int(np.round(
                (end_freq - start_freq) / sample_spacing)) + 1
            return np.linspace(start_freq, end_freq, num_samples)
        except Exception:
            LOGGER.error('Failed to calculate frequency points to sweep over a test channel')


def get_dsim_source_info(dsim):
    """Return a dict with all the current sine, noise and output settings of a dsim"""
    info = dict(sin_sources={}, noise_sources={}, outputs={})
    for sin_src in dsim.sine_sources:
        info['sin_sources'][sin_src.name] = dict(scale=sin_src.scale, frequency=sin_src.frequency)
    for noise_src in dsim.noise_sources:
        info['noise_sources'][noise_src.name] = dict(scale=noise_src.scale)

    for output in dsim.outputs:
        info['outputs'][output.name] = dict(output_type=output.output_type, scale=output.scale)
    return info


def iterate_recursive_dict(dictionary, keys=()):
    """Generator; walk through a recursive dict structure

    yields the compound key and value of each leaf.

    Example
    =======
    ::
      eg = {
      'key_1': 'value_1',
      'key_2': {'key_21': 21,
              'key_22': {'key_221': 221, 'key_222': 222}}}

      for compound_key, val in iterate_recursive_dict(eg):
         print '{}: {}'.format(compound_key, val)

    should produce output:

    ::
      ('key_1', ): value_1
      ('key_2', 'key_21'): 21
      ('key_2', 'key_22', 'key_221'): 221
      ('key_2', 'key_22', 'key_222'): 222

    """
    if isinstance(dictionary, Mapping):
        for k in dictionary:
            for rv in iterate_recursive_dict(dictionary[k], keys + (k,)):
                yield rv
    else:
        yield (keys, dictionary)


def get_feng_snapshots(feng_fpga, timeout=5):
    snaps = {}
    for snap in feng_fpga.snapshots:
        snaps[snap.name] = snap.read(
            man_valid=False, man_trig=False, timeout=timeout)
    return snaps


def get_snapshots(instrument, timeout=60):
    try:
        f_snaps = threaded_fpga_operation(instrument.fhosts, timeout,
                                          (get_feng_snapshots,))
        return dict(feng=f_snaps)
    except Exception:
        return False


def rearrange_snapblock(snap_data, reverse=False):
    segs = []
    for segment in sorted(snap_data.keys(), reverse=reverse):
        segs.append(snap_data[segment])
    return np.column_stack(segs).flatten()


def get_quant_snapshot(self, input_name, timeout=5):
    """Get the quantiser snapshot of named input. Snapshot will be assembled"""
    data_sources = [_source
                    for _input, _source in self.correlator.fengine_sources.iteritems()
                    if input_name == _input][0]

    snap_name = 'snap_quant{}_ss'.format(data_sources.source_number)
    snap = data_sources.host.snapshots[snap_name]
    snap_data = snap.read(
        man_valid=False, man_trig=False, timeout=timeout)['data']

    def get_part(qd, part):
        return {k: v for k, v in qd.items() if k.startswith(part)}

    real = rearrange_snapblock(get_part(snap_data, 'real'))
    imag = rearrange_snapblock(get_part(snap_data, 'imag'))
    quantiser_spectrum = real + 1j * imag
    return quantiser_spectrum


def get_baselines_lookup(self, test_input=None, auto_corr_index=False, sorted_lookup=False):
    """Get list of all the baselines present in the correlator output.
    Param:
        self: object
        Return: dict:
        baseline lookup with tuple of input label strings keys
        `(bl_label_A, bl_label_B)` and values bl_AB_ind, the index into the
        correlator dump's baselines
    """
    try:
        bls_ordering = eval(self.cam_sensors.get_value('bls_ordering'))
        LOGGER.info('Retrieved bls ordering via CAM Interface')
    except Exception:
        bls_ordering = None
        LOGGER.exception('Failed to retrieve bls ordering from CAM int.')
        return

    baseline_lookup = {tuple(bl): ind for ind, bl in enumerate(bls_ordering)}
    if auto_corr_index:
        for idx, val in enumerate(baseline_lookup):
            if val[0] == test_input and val[1] == test_input:
                auto_corr_idx = idx
        return [baseline_lookup, auto_corr_idx]
    elif sorted_lookup:
        return sorted(baseline_lookup.items(), key=operator.itemgetter(1))
    else:
        return baseline_lookup


def clear_all_delays(self, num_int=10):
    """Clears all delays on all fhosts.
    Param: object
    Param: num_int: Number of integrations for calculation time to apply delays and number of
                    spead accumulation discards.
    Return: Boolean
    """
    no_fengines = self.cam_sensors.get_value('n_fengs')
    int_time = self.cam_sensors.get_value('int_time')
    LOGGER.info('Clearing all delays and phases')

    delay_coefficients = ['0,0:0,0'] * no_fengines
    _retries = 10
    errmsg = ''
    _delays_set = False
    while _retries:
        _retries -= 1
        try:
            if not _delays_set:
                dump = self.receiver.get_clean_dump(discard=0)
                dump_timestamp = dump['dump_timestamp']
                time_now = time.time()
                time_diff = dump_timestamp - time_now
                errmsg = ('Dump timestamp (%s) is not in-sync with epoch (%s) [diff: %s]' % (
                    dump_timestamp, time_now, time_diff))
                # assert int(time_diff) in [-1, 0], errmsg
                t_apply = dump_timestamp + (num_int * int_time)
                reply, informs = self.corr_fix.katcp_rct.req.delays(t_apply, *delay_coefficients)
                errmsg = 'Delays command could not be executed in the given time'
                assert reply.reply_ok(), errmsg
                _delays_set = reply.reply_ok()
                time.sleep(5)

            dump = self.receiver.get_clean_dump()
            _max = int(np.max(np.angle(dump['xeng_raw'][:, 0,:][10:-10])))
            _min = int(np.min(np.angle(dump['xeng_raw'][:, 0,:][10:-10])))
            errmsg = 'Max(%s)/Min(%s) delays found, not cleared' % (_max, _min)
            assert _min ==_max == 0, errmsg
            LOGGER.info(
                'Delays cleared successfully. Dump timestamp is in-sync with epoch: %s' % time_diff)
            return True
        except AssertionError:
            LOGGER.warning(errmsg + ', within %s retries.' % _retries)
        except TypeError:
            LOGGER.exception("Object has no attributes")
            return False
        except Exception:
            LOGGER.exception(errmsg + ', within %s retries.' % _retries)
    return False



def get_vacc_offset(xeng_raw):
    """Assuming a tone was only put into input 0,
       figure out if VACC is rooted by 1"""
    input0 = np.abs(complexise(xeng_raw[:, 0]))
    input1 = np.abs(complexise(xeng_raw[:, 1]))
    if np.max(input0) > float(0) and np.max(input1) == float(0):
        # We expect autocorr in baseline 0 to be nonzero if the vacc is
        # properly aligned, hence no offset
        return 0
    elif np.max(input1) > float(0) and np.max(input0) == float(0):
        return 1
    else:
        return False


def get_and_restore_initial_eqs(self):
    """ Retrieve input gains/eq and added clean-up to restore eq's in case their altered
    :param: self
    :rtype: dict
    """
    try:
        reply, informs = self.corr_fix.katcp_rct.req.gain_all()
        assert reply.reply_ok()
    except Exception:
        LOGGER.exception('Failed to retrieve gains via CAM int.')
        return
    else:
        input_labels = self.cam_sensors.input_labels
        gain = reply.arguments[-1]
        initial_equalisations = {}
        for label in input_labels:
            initial_equalisations[label] = gain

    def restore_initial_equalisations():
        try:
            init_eq = ''.join(list(set(initial_equalisations.values())))
            reply, informs = self.corr_fix.katcp_rct.req.gain_all(init_eq, timeout=cam_timeout)
            assert reply.reply_ok()
            return True
        except Exception:
            msg = 'Failed to set gain for all inputs with gain of %s' % init_eq
            LOGGER.exception(msg)
            return False

    self.addCleanup(restore_initial_equalisations)
    return initial_equalisations


def get_bit_flag(packed, flag_bit):
    flag_mask = 1 << flag_bit
    flag = bool(packed & flag_mask)
    return flag


def get_set_bits(packed, consider_bits=None):
    packed = int(packed)
    set_bits = set()
    for bit in range(packed.bit_length()):
        if get_bit_flag(packed, bit):
            set_bits.add(bit)
    if consider_bits is not None:
        set_bits = set_bits.intersection(consider_bits)
    return set_bits


def get_pfb_counts(status_dict):
    """Checks if FFT overflows and QDR status on roaches
    Param: fhosts items
    Return: Dict:
        Dictionary with pfb counts
    """
    pfb_list = {}
    for host, pfb_value in status_dict:
        pfb_list[host] = (pfb_value['pfb_of0_cnt'], pfb_value['pfb_of1_cnt'])
    return pfb_list


def get_adc_snapshot(fpga):
    data = fpga.get_adc_snapshots()
    rv = {'p0': [], 'p1': []}
    for ctr in range(0, len(data['p0']['d0'])):
        for ctr2 in range(0, 8):
            rv['p0'].append(data['p0']['d%i' % ctr2][ctr])
            rv['p1'].append(data['p1']['d%i' % ctr2][ctr])
    return rv


def set_default_eq(self):
    """ Iterate through config sources and set eq's as per config file
    Param: Correlator: Object
    Return: None
    """
    def get_eqs_levels(self):
        eq_levels = []
        try:
            for eq_label in [i for i in self.correlator.configd['fengine'] if i.startswith('eq')]:
                eq_levels.append(complex(self.correlator.configd['fengine'][eq_label]))
            return eq_levels
        except Exception:
            LOGGER.exception('Failed to retrieve default ant_inputs and eq levels from config file')
            return False

    eq_levels = list(set(get_eqs_levels(self)))[0]
    try:
        assert isinstance(eq_levels, complex)
        reply, informs = self.corr_fix.katcp_rct.req.gain_all(eq_levels, timeout=cam_timeout)
        assert reply.reply_ok()
        LOGGER.info('Reset gains to default values from config file.\n')
        return True
    except Exception:
        LOGGER.exception('Failed to set gains on all inputs with %s ' % (eq_levels))
        return False


def set_input_levels(self, awgn_scale=None, cw_scale=None, freq=None, fft_shift=None, gain=None,
                     cw_src=0):
    """
    Set the digitiser simulator (dsim) output levels, FFT shift
    and quantiser gain to optimum levels - Hardcoded.
    Param:
        self: Object
            correlator_fixture object
        awgn_scale : Float
            Gaussian noise digitiser output scale.
        cw_scale: Float
            constant wave digitiser output scale.
        freq: Float
            arbitrary frequency to set with the digitiser simulator
        fft_shift: Int
            current FFT shift value
        gain: Complex/Str
            quantiser gain value
        cw_src: Int
            source 0 or 1
    Return: Bool
    """
    if cw_scale is not None:
        self.dhost.sine_sources.sin_0.set(frequency=freq, scale=cw_scale)
        self.dhost.sine_sources.sin_1.set(frequency=freq, scale=cw_scale)

    if awgn_scale is not None:
        self.dhost.noise_sources.noise_corr.set(scale=awgn_scale)

    def set_fft_shift(self):
        try:
            reply, _informs = self.corr_fix.katcp_rct.req.fft_shift(fft_shift, timeout=cam_timeout)
            assert reply.reply_ok()
            LOGGER.info('F-Engines FFT shift set to {} via CAM interface'.format(fft_shift))
            return True
        except Exception as e:
            errmsg = 'Failed to set FFT shift via CAM interface due to %s' %str(e)
            LOGGER.exception(errmsg)
            return False

    LOGGER.info('Setting desired FFT-Shift via CAM interface.')
    if set_fft_shift(self) is not True:
        LOGGER.error('Failed to set FFT-Shift via CAM interface')

    sources = self.cam_sensors.input_labels
    source_gain_dict = dict(ChainMap(*[{i: '{}'.format(gain)} for i in sources]))
    try:
        LOGGER.info('Setting desired gain/eq via CAM interface.')
        eq_level = list(set(source_gain_dict.values()))
        if len(eq_level) is not 1:
            for i, v in source_gain_dict.items():
                LOGGER.info('Input %s gain set to %s' % (i, v))
                reply, informs = self.corr_fix.katcp_rct.req.gain(i, v, timeout=cam_timeout)
                assert reply.reply_ok()
        else:
            eq_level = eq_level[0]
            LOGGER.info('Setting gain levels to all inputs to %s' % (eq_level))
            reply, informs = self.corr_fix.katcp_rct.req.gain_all(eq_level, timeout=cam_timeout)
            assert reply.reply_ok()
        LOGGER.info('Gains set successfully: Reply:- %s' %str(reply))
        return True
    except Exception as e:
        errmsg = 'Failed to set gain for input due to %s' %str(e)
        LOGGER.exception(errmsg)
        return False


def get_delay_bounds(correlator):
    """
    Parameters
    ----------
    correlator - As displayed in you on board flight manual

    Returns
    -------
    Dictionary containing minimum and maximum values for delay, delay rate,
    phase offset and phase offset rate

    """
    fhost = correlator.fhosts[0]
    # Get maximum delay value
    reg_info = fhost.registers.delay0.block_info
    reg_bw = int(reg_info['bitwidths'])
    reg_bp = int(reg_info['bin_pts'])
    max_delay = 2 ** (reg_bw - reg_bp) - 1 / float(2 ** reg_bp)
    max_delay = max_delay / correlator.sample_rate_hz
    min_delay = 0
    # Get maximum delay rate value
    reg_info = fhost.registers.delta_delay0.block_info
    _b = int(reg_info['bin_pts'])
    max_positive_delta_delay = 1 - 1 / float(2 ** _b)
    max_negative_delta_delay = -1 + 1 / float(2 ** _b)
    # Get max/min phase offset
    reg_info = fhost.registers.phase0.block_info
    b_str = reg_info['bin_pts']
    _b = int(b_str[1: len(b_str) - 1].rsplit(' ')[0])
    max_positive_phase_offset = 1 - 1 / float(2 ** _b)
    max_negative_phase_offset = -1 + 1 / float(2 ** _b)
    max_positive_phase_offset *= float(np.pi)
    max_negative_phase_offset *= float(np.pi)
    # Get max/min phase rate
    b_str = reg_info['bin_pts']
    _b = int(b_str[1: len(b_str) - 1].rsplit(' ')[1])
    max_positive_delta_phase = 1 - 1 / float(2 ** _b)
    max_negative_delta_phase = -1 + 1 / float(2 ** _b)
    # As per fhost_fpga
    bitshift_schedule = 23
    bitshift = (2 ** bitshift_schedule)
    max_positive_delta_phase = (max_positive_delta_phase * float(np.pi) *
                                correlator.sample_rate_hz) / bitshift
    max_negative_delta_phase = (max_negative_delta_phase * float(np.pi) *
                                correlator.sample_rate_hz) / bitshift
    return {
            'max_delay_value': max_delay,
            'min_delay_value': min_delay,
            'max_delay_rate': max_positive_delta_delay,
            'min_delay_rate': max_negative_delta_delay,
            'max_phase_offset': max_positive_phase_offset,
            'min_phase_offset': max_negative_phase_offset,
            'max_phase_rate': max_positive_delta_phase,
            'min_phase_rate': max_negative_delta_phase
            }


def disable_warnings_messages(spead2_warn=True, corr_warn=True, casperfpga_debug=True,
                              plt_warn=True, np_warn=True, deprecated_warn=True, katcp_warn=True):
    """This function disables all error warning messages
    :param:
        spead2 : Boolean
        corr_rx : Boolean
        plt : Boolean
        np : Boolean
        deprecated : Boolean
    :rtype: None
    """
    if spead2_warn:
        logging.getLogger('spead2').setLevel(logging.INFO)
    if corr_warn:
        logging.getLogger("corr2.corr_rx").setLevel(logging.DEBUG)
        logging.getLogger('corr2.xhost_fpga').setLevel(logging.CRITICAL)
        logging.getLogger('corr2.fhost_fpga').setLevel(logging.CRITICAL)
        logging.getLogger('fxcorrelator_fengops').setLevel(logging.CRITICAL)
        logging.getLogger('fhost_fpga').setLevel(logging.CRITICAL)
    if casperfpga_debug:
        logging.getLogger("casperfpga.bitfield").setLevel(logging.CRITICAL)
        logging.getLogger("casperfpga.katcp_fpga").setLevel(logging.CRITICAL)
        logging.getLogger("casperfpga.memory").setLevel(logging.CRITICAL)
        logging.getLogger("casperfpga.register ").setLevel(logging.CRITICAL)
        logging.getLogger("casperfpga.transport_katcp").setLevel(logging.CRITICAL)
    if katcp_warn:
        logging.getLogger('katcp').setLevel(logging.CRITICAL)
        logging.getLogger('tornado.application').setLevel(logging.CRITICAL)
    if plt_warn:
        import matplotlib
        warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
    if np_warn:
        # Ignoring all warnings raised when casting a complex dtype to a real dtype.
        warnings.simplefilter("ignore", np.ComplexWarning)
    if deprecated_warn:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class Text_Style(object):
    """Text manipulation"""

    def __init__(self):
        self.BOLD = '\033[1m'
        self.UNDERLINE = '\033[4m'
        self.END = '\033[0m'
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RED = '\033[91m'

    def Red(self, msg=None):
        return (self.RED + msg + self.END)

    def Green(self, msg=None):
        return (self.GREEN + msg + self.END)

    def Yellow(self, msg=None):
        return (self.YELLOW + msg + self.END)

    def Bold(self, msg=None):
        return (self.BOLD + msg + self.END)

    def Underline(self, msg=None):
        return (self.UNDERLINE + msg + self.END)


Style = Text_Style()


@contextlib.contextmanager
def ignored(*exceptions):
    """
    Context manager to ignore specified exceptions
    :param: Exception
    :rtype: None
    """
    try:
        yield
    except exceptions:
        pass


class RetryError(Exception):
    pass


def retryloop(attempts, timeout):
    """
    Usage:

    for retry in retryloop(10, timeout=30):
        try:
            something
        except SomeException:
            retry()

    for retry in retryloop(10, timeout=30):
        something
        if somecondition:
            retry()

    """
    starttime = time.time()
    success = set()
    for i in range(attempts):
        success.add(True)
        yield success.clear
        if success:
            return
        if starttime + timeout < time.time():
            break
    raise RetryError('Failed to after %s attempts' % attempts)


def clear_host_status(self, timeout=60):
    """Clear the status registers and counters on this host
    :param: Self (Object):
    :param: timeout: Int
    :rtype: Boolean
    """
    try:
        hosts = self.correlator.fhosts + self.correlator.xhosts
        threaded_fpga_function(hosts, timeout, 'clear_status')
        LOGGER.info('Cleared status registers and counters on all F/X-Engines.')
        return True
    except Exception:
        LOGGER.error('Failed to clear status registers and counters on all F/X-Engines.')
        return False

def restore_src_names(self):
    """Restore default CBF input/source names.
    :param: Object
    :rtype: Boolean
    """
    try:
        orig_src_names = self.correlator.configd['fengine']['source_names'].split(',')
    except Exception:
        orig_src_names = ['ant_{}'.format(x) for x in xrange(self.correlator.n_antennas * 2)]

    LOGGER.info('Restoring source names to %s' % (', '.join(orig_src_names)))
    try:
        reply, informs = self.corr_fix.katcp_rct.req.input_labels(*orig_src_names)
        assert reply.reply_ok()
        self.corr_fix.issue_metadata
        LOGGER.info('Successfully restored source names back to default %s' % (', '.join(
            orig_src_names)))
    except Exception:
        LOGGER.exception('Failed to restore CBF source names back to default.')
        return False


def blindwrite(host):
    """Writes junk to host memory
    :param: host object
    :rtype: Boolean
    """
    junk_msg = ('0x' + ''.join(x.encode('hex') for x in 'oidhsdvwsfvbgrfbsdceijfp3ioejfg'))
    try:
        for i in xrange(1000):
            host.blindwrite('qdr0_memory', junk_msg)
            host.blindwrite('qdr1_memory', junk_msg)
        LOGGER.info('Wrote junk chars to QDR\'s memory on host: %s.'% host.host)
        return True
    except:
        LOGGER.error('Failed to write junk chars to QDR\'s memory on host: %s.'% host.host)
        return False


def human_readable_ip(hex_ip):
    hex_ip = hex_ip[2:]
    return '.'.join([str(int(x + y, 16)) for x, y in zip(hex_ip[::2], hex_ip[1::2])])


def confirm_out_dest_ip(self):
    """Confirm is correlators output destination ip is the same as the one in config file
    :param: Object
    :rtype: Boolean
    """
    parse_address = StreamAddress._parse_address_string
    try:
        xhost = self.correlator.xhosts[random.randrange(len(self.correlator.xhosts))]
        int_ip = int(xhost.registers.gbe_iptx.read()['data']['reg'])
        xhost_ip = inet_ntoa(pack(">L", int_ip))
        dest_ip = list(parse_address(
            self.correlator.configd['xengine']['output_destinations_base']))[0]
        assert dest_ip == xhost_ip
        return True
    except Exception:
        LOGGER.exception('Failed to retrieve correlator ip address.')
        return False


class TestTimeout:
    """
    Test Timeout class using ALARM signal.
    :param: seconds -> Int
    :param: error_message -> Str
    :rtype: None
    """

    class TestTimeoutError(Exception):
        """Custom TestTimeoutError exception"""
        pass

    def __init__(self, seconds=1, error_message='Test Timed-out'):
        self.seconds = seconds
        self.error_message = ''.join([error_message, ' after {} seconds'.format(self.seconds)])

    def handle_timeout(self, signum, frame):
        raise TestTimeout.TestTimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


@contextlib.contextmanager
def RunTestWithTimeout(test_timeout, errmsg='Test Timed-out'):
    """
    Context manager to execute tests with a timeout
    :param: test_timeout : int
    :param: errmsg : str
    :rtype: None
    """
    try:
        with TestTimeout(seconds=test_timeout):
            yield
    except Exception:
        LOGGER.exception(errmsg)
        Aqf.failed(errmsg)
        Aqf.end(traceback=True)


def executed_by():
    """Get who ran the test."""
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
        if user == 'root':
            raise OSError
        Aqf.hop('Test ran by: {} on {} system on {}.\n'.format(user, os.uname()[1].upper(),
                                                               time.strftime("%Y-%m-%d %H:%M:%S")))
    except Exception as e:
        _errmsg = 'Failed to detemine who ran the test with %s' % str(e)
        LOGGER.error(_errmsg)
        Aqf.hop('Test ran by: Jenkins on system {} on {}.\n'.format(os.uname()[1].upper(),
                                                                    time.ctime()))

def encode_passwd(pw_encrypt, key=None):
    """This function encrypts a string with base64 algorithm
    :param: pw_encrypt: Str

    :param: secret key : Str = 16 chars long
        Keep your secret_key safe!
    :rtype: encrypted password
    """
    if key is not None:
        _cipher = AES.new(key, AES.MODE_ECB)
        encoded = base64.b64encode(_cipher.encrypt(pw_encrypt.rjust(32)))
        return encoded

def decode_passwd(pw_decrypt, key=None):
    """This function decrypts a string with base64 algorithm
    :param: pw_decrypt: Str

    :param: secret key : Str = 16 chars long
        Keep your secret_key safe!
    :rtype: decrypted password
    """
    if key is not None:
        _cipher = AES.new(key, AES.MODE_ECB)
        decoded = _cipher.decrypt(base64.b64decode(pw_decrypt))
        return decoded.strip()

def create_logs_directory(self):
    """
    Create custom `logs_instrument` directory on the test dir to store generated images
    param: self
    """
    test_dir, test_name = os.path.split(os.path.dirname(
                os.path.realpath(__file__)))
    path = test_dir + '/logs_' + self.corr_fix.instrument
    if not os.path.exists(path):
        LOGGER.info('Created %s for storing images.'% path)
        os.makedirs(path)
    return path


class GetSensors(object):
    """Easily get sensor values without much work"""
    def __init__(self, corr_fix):
        self.req = corr_fix.katcp_rct.req
        self.sensors = corr_fix.katcp_rct.sensor

    def get_value(self, _name):
        """
        Get sensor Value(s)

        Parameters
        ----------
        str: sensor name e.g. n_bls

        Return
        ---------
        List or Str or None: sensor value"""
        if any(_name in s for s in dir(self.sensors)):
            _attribute = [s for s in dir(self.sensors) if _name in s][0]
            return getattr(self.sensors, _attribute).get_value()

    @property
    def input_labels(self):
        """
        Simplified input labels(s)

        Return
        ---------
        List: simplified input labels
        """
        try:
            input_labelling = eval(self.sensors.input_labelling.get_value())
            input_labels = [x[0] for x in [list(i) for i in input_labelling]]
        except Exception:
            input_labels = str(self.req.input_labels()).split()[2:]
        return input_labels

    @property
    def custom_input_labels(self):
        """
        Simplified custom input labels(s)

        Return
        ---------
        List: simplified custom input labels
        """
        n_ants = int(self.get_value('n_ants'))
        return ['inp0{:02d}_{}'.format(x, i) for x in xrange(n_ants) for i in 'xy']

    @property
    def ch_center_freqs(self):
        """
        Calculates the center frequencies of all channels.
        First channel center frequency is 0.
        Second element can be used as the channel bandwidth

        Return
        ---------
        List: channel center frequencies
        """
        n_chans = float(self.get_value('n_chans'))
        bandwidth = float(self.get_value('bandwidth'))
        ch_bandwidth = bandwidth / n_chans
        f_start = 0. # Center freq of the first channel
        return f_start + np.arange(n_chans) * ch_bandwidth

    @property
    def sample_period(self):
        """
        Get sample rate and return sample period
        """
        return 1/float(self.get_value('adc_sample_rate'))

    @property
    def fft_period(self):
        """
        Get FFT Period
        """
        return self.sample_period * 2 * float(self.get_value('n_chans'))

    @property
    def delta_f(self):
        """
        Get Correlator bandwidth
        """
        return float(self.get_value('bandwidth')/self.get_value('n_chans'))

    def calc_freq_samples(self, chan, samples_per_chan, chans_around=0):
        """Calculate frequency points to sweep over a test channel.

        Parameters
        =========
        chan : int
           Channel number around which to place frequency samples
        samples_per_chan: int
           Number of frequency points per channel
        chans_around: int
           Number of channels to include around the test channel. I.e. value 1 will
           include one extra channel above and one below the test channel.

        Will put frequency sample on channel boundary if 2 or more points per channel are
        requested, and if will place a point in the centre of the channel if an odd number
        of points are specified.

        """
        assert samples_per_chan > 0
        assert chans_around > 0
        assert 0 <= chan < self.get_value('n_chans')
        assert 0 <= chan + chans_around < self.get_value('n_chans')
        assert 0 <= chan - chans_around < self.get_value('n_chans')

        start_chan = chan - chans_around
        end_chan = chan + chans_around
        if samples_per_chan == 1:
            return self.ch_center_freqs[start_chan:end_chan + 1]
        start_freq = self.ch_center_freqs[start_chan] - self.delta_f / 2
        end_freq = self.ch_center_freqs[end_chan] + self.delta_f / 2
        sample_spacing = self.delta_f / (samples_per_chan - 1)
        num_samples = int(np.round((end_freq - start_freq) / sample_spacing)) + 1
        return np.linspace(start_freq, end_freq, num_samples)



def start_katsdpingest_docker(self, beam_ip, beam_port, partitions, channels=4096,
                              ticks_between_spectra=8192, channels_per_heap=256, spectra_per_heap=256):
    """ Starts a katsdpingest docker containter. Kills any currently running instances.

    Returns
    -------
        False if katsdpingest docer not started
        True if katsdpingest docker started
    """
    user_id = pwd.getpwuid(os.getuid()).pw_uid
    cmd = ['docker', 'run', '-u', '{}'.format(user_id), '-d', '--net=host', '-v',
           '/ramdisk:/ramdisk', 'sdp-docker-registry.kat.ac.za:5000/katsdpingest:cbf_testing',
           'bf_ingest.py', '--cbf-spead={}+{}:{} '.format(beam_ip, partitions-1,beam_port),
           '--channels={}'.format(channels), '--ticks-between-spectra={}'.format(ticks_between_spectra),
           '--channels-per-heap={}'.format(channels_per_heap), '--spectra-per-heap={}'.format(
            spectra_per_heap), '--file-base=/ramdisk', '--log-level=DEBUG']

    stop_katsdpingest_docker(self)
    try:
        LOGGER.info('Executing docker command to run KAT SDP injest node')
        output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError:
        errmsg = ('Could not start sdp-docker-registry container, '
                  'ensure SDP Ingest image has been built successfully')
        Aqf.failed(errmsg)
        LOGGER.exception(errmsg)
        return False

    time.sleep(5)
    try:
        output = subprocess.check_output(['docker', 'ps'])
    except subprocess.CalledProcessError:
        return False
    output = output.split()
    sdp_instance = [idx for idx,s in enumerate(output) if 'sdp-docker-registry.kat.ac.za' in s]
    # If sdp-docker-registry not found it is not running, return false
    if sdp_instance:
        return True
    else:
        return False

def stop_katsdpingest_docker(self):
    """ Finds if a katsdpingest docker containter is running and kills it.

    Returns
    -------
        False if katsdpingest docker container not found or not running
        True if katsdpingest docker container found and stopped
    """
    try:
        output = subprocess.check_output(['docker','ps'])
    except subprocess.CalledProcessError:
        return False
    output = output.split()
    sdp_instance = [idx for idx,s in enumerate(output) if 'sdp-docker-registry.kat.ac.za' in s]
    # If sdp-docker-registry not found it is not running, return false
    # Kill all instances found
    if sdp_instance:
        for idx in sdp_instance:
            try:
                kill_output = subprocess.check_output(['docker', 'kill', output[idx-1]])
            except subprocess.CalledProcessError:
                errmsg = 'Could not kill sdp-docker-registry container'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False
            killed_id = kill_output.split()[0]
            if killed_id != output[idx-1]:
                errmsg = 'Could not kill sdp-docker-registry container'
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
                return False

    else:
        return False
    return True

def capture_beam_data(self, beam, beam_dict, ingest_kcp_client=None, capture_time=0.1):
    """ Capture beamformer data

    Parameters
    ----------
    beam (beam_0x, beam_0y):
        Polarisation to capture beam data
    beam_dict:
        Dictionary containing input:weight key pairs e.g.
        beam_dict = {'m000_x': 1.0, 'm000_y': 1.0}
        If beam_dict = None weights will not be set
    ingest_kcp_client:
        katcp client for ingest node, if None one will be created.
    capture_time:
        Number of seconds to capture beam data

    Returns
    -------
        bf_raw:
            Raw beamformer data for the selected beam
        cap_ts:
            Captured timestamps, dropped packet timestamps will not be
            present
        bf_ts:
            Expected timestamps

    """
    beamdata_dir = '/ramdisk'
    _timeout = 10
    
    import katcp
    # Create a katcp client to connect to katcpingest if one not specified
    if ingest_kcp_client == None:
        if os.uname()[1] == 'cmc2':
            ingst_nd = self.corr_fix._test_config_file['beamformer']['ingest_node_cmc2']
        elif os.uname()[1] == 'cmc3':
            ingst_nd = self.corr_fix._test_config_file['beamformer']['ingest_node_cmc3']
        else:
            ingst_nd = self.corr_fix._test_config_file['beamformer']['ingest_node']
        ingst_nd_p = self.corr_fix._test_config_file['beamformer']['ingest_node_port']
        try:
            ingest_kcp_client = katcp.BlockingClient(ingst_nd, ingst_nd_p)
            ingest_kcp_client.setDaemon(True)
            ingest_kcp_client.start()
            self.addCleanup(ingest_kcp_client.stop)
            is_connected = ingest_kcp_client.wait_connected(_timeout)
            if not is_connected:
                errmsg = 'Could not connect to %s:%s, timed out.' %(ingst_nd, ingst_nd_p)
                ingest_kcp_client.stop()
                raise RuntimeError(errmsg)
        except Exception as e:
            LOGGER.exception(str(e))
            Aqf.failed(str(e))

    # Build new dictionary with only the requested beam keys:value pairs
    in_wgts = {}
    beam_pol = beam[-1]
    if beam_dict:
        for key in beam_dict:
            if key.find(beam_pol) != -1:
                in_wgts[key] = beam_dict[key]

        Aqf.step('Setting input weights, this may take a long time, check log output for progress...')
        print_list = ''
        for key in in_wgts:
            LOGGER.info('Confirm that antenna input ({}) weight has been set to the desired weight.'.format(
                key))
            try:
                reply, informs = self.corr_fix.katcp_rct.req.beam_weights(beam, key, in_wgts[key])
                assert reply.reply_ok()
            except AssertionError:
                Aqf.failed('Beam weights not successfully set')
            except Exception as e:
                errmsg = 'Test failed due to %s'%str(e)
                Aqf.failed(errmsg)
                LOGGER.exception(errmsg)
            else:
                LOGGER.info('Antenna input {} weight set to {}'.format(key, reply.arguments[1]))
                print_list += ('{}:{}, '.format(key,reply.arguments[1]))
                in_wgts[key] = float(reply.arguments[1])
        Aqf.passed('Antenna input weights set to: {}'.format(print_list[:-2]))

    try:
        LOGGER.info('Issue {} capture start via CAM int'.format(beam))
        for i in xrange(2):
            reply, informs = self.corr_fix.katcp_rct.req.capture_meta(beam)
        errmsg = 'Failed to issue new Metadata: {}'.format(str(reply))
        assert reply.reply_ok(), errmsg
        reply, informs = self.corr_fix.katcp_rct.req.capture_start(beam)
        errmsg = 'Failed to issue capture_start for beam {}: {}'.format(beam,str(reply))
        assert reply.reply_ok(), errmsg
    except AssertionError:
        errmsg = ' .'.join([errmsg, 'Failed to start Data transmission.'])
        Aqf.failed(errmsg)
    try:
        LOGGER.info('Issue ingest node capture-init.')
        reply, informs = ingest_kcp_client.blocking_request(katcp.Message.request('capture-init'),
            timeout=_timeout)
        errmsg = 'Failed to issues ingest node capture-init: {}'.format(str(reply))
        assert reply.reply_ok(), errmsg
    except Exception as e:
        print e
        LOGGER.exception(e)
        LOGGER.exception(errmsg)
        Aqf.failed(errmsg)
    LOGGER.info('Capturing beam data for {} seconds'.format(capture_time))
    time.sleep(capture_time)
    try:
        LOGGER.info('Issue ingest node capture-done.')
        reply, informs = ingest_kcp_client.blocking_request(katcp.Message.request('capture-done'),
            timeout=_timeout)
        errmsg = 'Failed to issues ingest node capture-done: {}'.format(str(reply))
        assert reply.reply_ok(), errmsg
    except Exception:
        Aqf.failed(errmsg)
        LOGGER.error(errmsg)
    try:
        LOGGER.info('Issue {} capture stop via CAM int'.format(beam))
        reply, informs = self.corr_fix.katcp_rct.req.capture_stop(beam)
        errmsg = 'Failed to issue capture_stop for beam {}: {}'.format(beam,str(reply))
        assert reply.reply_ok(), errmsg
    except AssertionError:
        Aqf.failed(errmsg)
        LOGGER.exception(errmsg)

    try:
        LOGGER.info('Getting latest beam data captured in %s'%beamdata_dir)
        newest_f = max(glob.iglob('%s/*.h5'%beamdata_dir), key=os.path.getctime)
        _timestamp = int(newest_f.split('/')[-1].split('.')[0])
        newest_f_timestamp = time.strftime("%H:%M:%S", time.localtime(_timestamp))
    except ValueError as e:
        Aqf.failed('Failed to get the latest beamformer data: %s'%str(e))
        return
    else:
        LOGGER.info('Reading h5py data file(%s)[%s] and extracting the beam data.\n'%(newest_f,
            newest_f_timestamp))
        with h5py.File(newest_f, 'r') as fin:
            data = fin['Data'].values()
            for element in data:
                # if element.name.find('captured_timestamps') > -1:
                #     cap_ts = np.array(element.value)
                if element.name.find('bf_raw') > -1:
                    bf_raw = np.array(element.value)
                elif element.name.find('timestamps') > -1:
                    bf_ts = np.array(element.value)
                elif element.name.find('flags') > -1:
                    bf_flags = np.array(element.value)
        os.remove(newest_f)
        return bf_raw, bf_flags, bf_ts, in_wgts

def populate_beam_dict(self, num_wgts_to_set, value, beam_dict):
    """
        If num_wgts_to_set = -1 all inputs will be set
    """
    ctr = 0
    for key in beam_dict:
        if ctr < num_wgts_to_set or num_wgts_to_set == -1:
            beam_dict[key] = value
            ctr += 1
    return beam_dict

def populate_beam_dict_idx(self, index, value, beam_dict):
    """
        Set specified beam index to weight, all other values to 0
    """
    for key in beam_dict:
        key_idx = int(filter(str.isdigit, key))
        if key_idx == index:
            beam_dict[key] = value
        else:
            beam_dict[key] = 0
    return beam_dict

def set_beam_quant_gain(self, beam, gain):
    try:
        reply, informs = self.corr_fix.katcp_rct.req.beam_quant_gains(beam, gain)
        assert reply.reply_ok()
        actual_beam_gain = float(reply.arguments[1])
        msg = ('Requested beamformer level adjust gain of {:.2f}, '
               'actual gain set to {:.2f}.'.format(gain, actual_beam_gain))
        Aqf.almost_equals(actual_beam_gain, gain, 0.1, msg)
        return actual_beam_gain
    except Exception, e:
        Aqf.failed('Failed to set beamformer quantiser gain via CAM interface, {}'.format(str(e)))
        return 0


class DictEval(object):
    """
     load variables in a dict into namespace
    """
    # Alt use (Not ideal): locals().update(adict)
    def __init__(self, adict):
        self.__dict__.update(adict)


def FPGA_Connect(hosts, _timeout=30):
    """Utility to connect to hosts via Casperfpga"""
    fpgas = False
    retry = 10
    while not fpgas:
        try:
            fpgas = threaded_create_fpgas_from_hosts(hosts, timeout=_timeout)
        except RuntimeError:
            retry -= 1
            if retry == 0:
                errmsg = 'ERROR: Could not connect to SKARABs'
                return False
    return fpgas

def get_clean_dump(self, retries=20):
    _captured = False
    while retries:
        retries -= 1
        try:
            errmsg = 'Queue is empty will retry (%s) ie EMPTY DUMPS!!!!!!!!!!!!!!!!!!!!!' % retries
            discards = 0
            if not _captured:
                dump = self.receiver.get_clean_dump(discard=discards)
                assert isinstance(dump, dict), errmsg
                _captured = True

            dump = self.receiver.data_queue.get(timeout=10)
            dump_timestamp = dump['dump_timestamp']
            dhost_timestamp = self.dhost.registers.sys_clkcounter.read().get('timestamp')
            time_diff = np.abs(dump_timestamp - dhost_timestamp)
            errmsg = ('Dump timestamp (%s) is not in-sync with digitiser sync epoch (%s) [diff: %s]' % (
                        dump_timestamp, dhost_timestamp, time_diff))
            if time_diff > 10:
                raise AssertionError
            # assert int(time_diff) in [-1, 0], errmsg
        except AssertionError:
            LOGGER.warning(errmsg)
        except Queue.Empty:
            errmsg = 'Could not retrieve clean SPEAD accumulation: Queue is Empty.'
            LOGGER.exception(errmsg)
            if retries < 15:
                return False
        else:
            LOGGER.info('Yeyyyyyyyyy: Dump timestamp (%s) in-sync with digitiser sync epoch (%s) [diff: %s] '
                        'within %s retries' % (dump_timestamp, dhost_timestamp, time_diff, retries))
            return dump



class CSV_Reader(object):
    """
    Manual Tests CSV reader

    Parameters
    ---------
        csv_filename: str, Valid path to csv file/url
        set_index: str, If you want to change the index, set name
    Returns
    -------
        result: Pandas DataFrame
    """
    def __init__(self, csv_filename, set_index=None):
        self.csv_filename = csv_filename
        self.set_index = set_index

    @property
    def load_csv(self):
        """
        Load csv file

        Parameters
        ----------
            object

        Returns
        -------
            result: Pandas DataFrame
        """
        try:
            assert self.csv_filename
            df = pd.read_csv(self.csv_filename)
            df = df.replace(np.nan, "TBD", regex=True)
            df = df.fillna(method='ffill')
        except:
            return False
        else:
            return df.set_index(self.set_index) if self.set_index else df

    def csv_to_dict(self, ve_number=None):
        """
        CSV contents to Dict

        Parameters
        ----------
            ve_number: Verification Event Number e.g. CBF.V.1.11

        Returns
        -------
        result: dict
        """
        return dict(self.load_csv.loc[ve_number]) if ve_number else None
