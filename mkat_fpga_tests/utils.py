import Queue
import base64
import contextlib
import logging
import os
import warnings

from collections import Mapping
from collections import defaultdict
from getpass import getuser as getusername
from random import randrange
from socket import inet_ntoa
from struct import pack
# MEMORY LEAKS DEBUGGING
# To use, add @DetectMemLeaks decorator to function
from memory_profiler import profile as DetectMemLeaks

import corr2
import signal
import time

from Crypto.Cipher import AES

import numpy as np

from nosekatreport import Aqf

try:
    from collections import ChainMap
except ImportError:
    from chainmap import ChainMap

from casperfpga.utils import threaded_fpga_function
from casperfpga.utils import threaded_fpga_operation
from inspect import currentframe
from inspect import getframeinfo


# LOGGER = logging.getLogger(__name__)
LOGGER = logging.getLogger('mkat_fpga_tests')

# Max range of the integers coming out of VACC
VACC_FULL_RANGE = float(2 ** 31)

cam_timeout = 60

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
    """Select dsim signal output, zero all sources, output scalings to 1

    Also clear noise diode and adc overrange flags
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
            try:
                sin_source.set(repeatN=0)
            except NotImplementedError:
                pass
            LOGGER.info('Digitiser simulator cw source %s reset to Zeros' %sin_source.name)
    except Exception:
        LOGGER.error('Failed to reset sine sources on dhost.')
        pass

    try:
        for noise_source in dhost.noise_sources:
            noise_source.set(scale=0)
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

    try:
        dhost.registers.cwg0_en.write(en=1)
        dhost.registers.cwg1_en.write(en=1)
        cwg0_en = dhost.registers.cwg0_en.read()['data']['en']
        cwg1_en = dhost.registers.cwg1_en.read()['data']['en']
        assert cwg0_en == cwg1_en
        LOGGER.info('Digitiser simulator cwg0 is enabled: %s' %cwg0_en)
        LOGGER.info('Digitiser simulator cwg1 is enabled: %s' %cwg1_en)
    except Exception:
        LOGGER.exception('Failed to enable dhost cwg0 and cwg1 registers.')
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
        except Exception:
            LOGGER.error('Failed to retrieve various bits of corr freq from corr config')


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


def get_quant_snapshot(instrument, input_name, timeout=5):
    """Get the quantiser snapshot of named input. Snapshot will be assembled"""
    data_sources = [_source
                    for _input, _source in instrument.fengine_sources.iteritems()
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


def get_baselines_lookup(spead):
    """Get list of all the baselines present in the correlator output.
    Param:
      spead: spead_dump
    Return: dict:
        baseline lookup with tuple of input label strings keys
        `(bl_label_A, bl_label_B)` and values bl_AB_ind, the index into the
        correlator dump's baselines
    """
    bls_ordering = spead['bls_ordering'].value
    baseline_lookup = {tuple(bl): ind for ind, bl in enumerate(bls_ordering)}
    return baseline_lookup

def get_sync_epoch(self):
    """ Get digitiser synch epoch
    :Param: object
    :rtype: float or boolean
    """
    try:
        reply, informs = self.corr_fix.katcp_rct.req.digitiser_synch_epoch()
        assert reply.reply_ok()
        return float(reply.arguments[-1])
    except Exception:
        errmsg = ('Could not retrieve sync time via correlator object.')
        LOGGER.exception(errmsg)
        Aqf.failed(errmsg)
        return False


def clear_all_delays(self, roundtrip=0.003, timeout=10):
    """Clears all delays on all fhosts.
    Param: object
    Return: Boolean
    """
    try:
        dump = self.receiver.get_clean_dump(timeout, discard=0)
    except Queue.Empty:
        LOGGER.exception('Could not retrieve clean SPEAD dump, as Queue is Empty. '
                         'Therefore Delays not cleared')
        return False
    else:
        sync_time = get_sync_epoch(self)
        if not sync_time:
            LOGGER.error('Digitiser sync epoch cannot be null')
            return

        dump_1_timestamp = (sync_time + roundtrip +
                            dump['timestamp'].value / dump['scale_factor_timestamp'].value)
        t_apply = dump_1_timestamp + 10 * dump['int_time'].value
        delay_coefficients = ['0,0:0,0'] * len(self.correlator.fengine_sources)
        try:
            reply = self.corr_fix.katcp_rct.req.delays(t_apply, *delay_coefficients)
            LOGGER.info('[CBF-REQ-0110] Cleared delays: %s' % (str(reply.reply.arguments[1])))
            return True
        except Exception:
            LOGGER.exception('Could not clear delays')
            return False


def get_fftoverflow_qdrstatus(correlator):
    """Get dict of all roaches present in the correlator
    Param: Correlator object
    Return: Dict:
        Roach, QDR status, PFB counts
    """
    fhosts = {}
    xhosts = {}
    dicts = {'fhosts': {}, 'xhosts': {}}
    fengs = correlator.fhosts
    xengs = correlator.xhosts
    for fhost in fengs:
        fhosts[fhost.host] = {}
        try:
            fhosts[fhost.host]['QDR_okay'] = fhost.qdr_okay()
        except Exception:
            return False
        for pfb, value in fhost.registers.pfb_ctrs.read()['data'].iteritems():
            fhosts[fhost.host][pfb] = value
        for xhost in xengs:
            xhosts[xhost.host] = {}
            try:
                xhosts[xhost.host]['QDR_okay'] = xhost.qdr_okay()
            except Exception:
                return False
    dicts['fhosts'] = fhosts
    dicts['xhosts'] = xhosts
    return dicts


def check_fftoverflow_qdrstatus(correlator, last_pfb_counts, status=False):
    """Checks if FFT overflows and QDR status on roaches
    Param: Correlator object, last known pfb counts
    Return: list:
        Roaches with QDR status errors
    """
    qdr_error_roaches = set()
    try:
        fftoverflow_qdrstatus = get_fftoverflow_qdrstatus(correlator)
    except Exception:
        return False
    if fftoverflow_qdrstatus is not False:
        curr_pfb_counts = get_pfb_counts(fftoverflow_qdrstatus['fhosts'].items())

    if curr_pfb_counts is not False:
        for (curr_pfb_host, curr_pfb_value), (curr_pfb_host_x, last_pfb_value) in zip(
                last_pfb_counts.items(), curr_pfb_counts.items()):
            if curr_pfb_host is curr_pfb_host_x:
                if curr_pfb_value != last_pfb_value:
                    if status:
                        Aqf.failed("PFB FFT overflow on {}".format(curr_pfb_host))

    for hosts_status in fftoverflow_qdrstatus.values():
        for host, _hosts_status in hosts_status.items():
            if _hosts_status['QDR_okay'] is False:
                if status:
                    Aqf.failed('QDR status on {} not Okay.'.format(host))
                qdr_error_roaches.add(host)

    return list(qdr_error_roaches)

def get_hosts_status(self, check_host_okay, list_sensor=None, engine_type=None, ):
            LOGGER.info('Retrieving %s sensors for %s.' %(list_sensor, engine_type.upper()))
            for _sensor in list_sensor:
                _status_hosts = check_host_okay(self, engine=engine_type, sensor=_sensor)
                if _status_hosts is not (True or None):
                    for _status in _status_hosts:
                        Aqf.failed(_status)
                        LOGGER.error('Failed :%s\nFile: %s line: %s' %(_status,
                             getframeinfo(currentframe()).filename.split('/')[-1],
                             getframeinfo(currentframe()).lineno))



def check_host_okay(self, engine=None, sensor=None):
    """
    Function retrieves PFB, LRU, QDR, PHY and reorder status on all F/X-Engines via Cam interface.
    :param: Object: self
    :param: Str: F/X-engine
    :param: Str: sensor
    :rtype: Boolean or List
    """
    try:
        reply, informs = self.corr_fix.katcp_rct.req.sensor_value(timeout=60)
        assert reply.reply_ok()
    except Exception:
        LOGGER.exception('Failed to retrieve sensor values via CAM interface.')
        return None
    else:
        if engine == 'feng':
            hosts = [_i.host.lower() for _i in self.correlator.fhosts]
        elif engine == 'xeng':
            hosts = [_i.host.lower() for _i in self.correlator.xhosts]
        else:
            LOGGER.error('Engine cannot be None')
            return None

        sensor_status = [[' '.join(i.arguments[2:]) for i in informs
                         if i.arguments[2] == '{}-{}-{}-ok'.format(host, engine, sensor)]
                         for host in hosts]
        _errors_list = []
        for i in sensor_status:
            try:
                assert int(i[0].split()[-1]) == 1
                return True
            except AssertionError:
                errmsg = '{} Failure/Error: {}'.format(sensor.upper(), i[0])
                LOGGER.error(errmsg)
                _errors_list.append(errmsg)
            except IndexError:
                LOGGER.fatal('The was an issue reading sensor-values via CAM interface, Investigate:'
                             'File: %s line: %s' % (
                                 getframeinfo(currentframe()).filename.split('/')[-1],
                                 getframeinfo(currentframe()).lineno))
                return None

        return _errors_list


def get_vacc_offset(xeng_raw):
    """Assuming a tone was only put into input 0,
       figure out if VACC is rooted by 1"""
    input0 = np.abs(complexise(xeng_raw.value[:, 0]))
    input1 = np.abs(complexise(xeng_raw.value[:, 1]))
    if np.max(input0) > 0 and np.max(input1) == 0:
        # We expect autocorr in baseline 0 to be nonzero if the vacc is
        # properly aligned, hence no offset
        return 0
    elif np.max(input1) > 0 and np.max(input0) == 0:
        return 1
    else:
        return False


def get_and_restore_initial_eqs(test_instance, correlator):
    initial_equalisations = correlator.fops.eq_get()

    def restore_initial_equalisations():
        try:
            for _input, _eq in initial_equalisations.iteritems():
                if type(_eq) is list: _eq = _eq[0]
                reply, informs = test_instance.corr_fix.katcp_rct.req.gain(_input, _eq,
                                                                           timeout=cam_timeout)
                assert reply.reply_ok()
            return True
        except Exception:
            msg = 'Failed to set gain for input %s with gain of %s' %(_input, _eq)
            LOGGER.exception(msg)
            return False

    test_instance.addCleanup(restore_initial_equalisations)
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

def get_input_labels(self):
    """
    :param: Obj: self
    :rtype: list: input labels
    """
    try:
        reply, _informs = self.corr_fix.katcp_rct.req.input_labels()
        assert reply.reply_ok()
    except Exception:
        errmsg = ('Failed to retrieve input lables via Cam interface with a reply: %s' % (str(reply)))
        LOGGER.error(errmsg)
        Aqf.failed(errmsg)
        return []
    else:
        return reply.arguments[1:]

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

    ant_inputs = get_input_labels(self)
    eq_levels = get_eqs_levels(self)
    if ant_inputs or eq_levels:
        try:
            for _input, _eq in zip(ant_inputs, eq_levels):
                reply, informs = self.corr_fix.katcp_rct.req.gain(_input, _eq, timeout=cam_timeout)
            assert reply.reply_ok()
            LOGGER.info('Reset gains to default values from config file.\n')
            return True
        except Exception:
            LOGGER.exception('Failed to set gains on %s with %s ' %(ant_inputs, eq_levels))
            return False
    else:
        LOGGER.error('Failed to retrieve %s or %s' %(ant_inputs, eq_levels))
        return False


def set_input_levels(self, awgn_scale=None, cw_scale=None, freq=None,
                     fft_shift=None, gain=None, cw_src=0):
    """
    Set the digitiser simulator (dsim) output levels, FFT shift
    and quantiser gain to optimum levels - Hardcoded.
    Param:
        self: Object
            correlator_fixture object
        awgn_scale : Float
            gaussian noise digitiser output scale.
        cw_scale: Float
            constant wave digitiser output scale.
        freq: Float
            abitrary frequency to set with the digitiser simulator
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
        except (TypeError, AssertionError):
            LOGGER.exception('Failed to set FFT shift via CAM interface.')
            return False

    if set_fft_shift(self) is not True:
        LOGGER.error('Failed to set FFT-Shift via CAM interface')

    sources = get_input_labels(self)
    source_gain_dict = dict(ChainMap(*[{i: '{}'.format(gain)} for i in sources]))
    try:
        for i, v in source_gain_dict.items():
            LOGGER.info('Input %s gain set to %s' % (i, v))
            reply, informs = self.corr_fix.katcp_rct.req.gain(i, v, timeout=cam_timeout)
            assert reply.reply_ok()
        LOGGER.info('Gains set successfully')
        return True
    except Exception:
        LOGGER.exception('Failed to set gain for input: %s with gain of: %s' %(i, v))
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
        'max_delay': max_delay,
        'min_delay': min_delay,
        'max_positive_delta_delay': max_positive_delta_delay,
        'max_negative_delta_delay': max_negative_delta_delay,
        'max_positive_phase_offset': max_positive_phase_offset,
        'max_negative_phase_offset': max_negative_phase_offset,
        'max_positive_delta_phase': max_positive_delta_phase,
        'max_negative_delta_phase': max_negative_delta_phase
    }


def get_figure_numbering(self):
    """
    Gets figure numbering from tests that are ran in alphabetical order.
    Param:
        self: Object
    Return: Dict
    """
    _test_name = 'test_{}'.format(self.corr_fix.instrument)
    fig_numbering = {y: str(x)
                     for x, y in enumerate([i
                                            for i in dir(self) if i.startswith(_test_name)], start=1)}

    def get_fig_prefix(version=None, _dict=fig_numbering):
        """
        Update the current figure numbering with a suffix depending on running
        instrument
        Param:
            version: int/float
            _dict: dict
        Return: dict
        """
        for key, value in _dict.items():
            _dict[key] = '{}.{}'.format(value, version)
        return _dict

    try:
        assert self.corr_freqs.n_chans == 4096
        return get_fig_prefix(1)
    except AssertionError:
        return get_fig_prefix(2)


def disable_warnings_messages(spead2_warn=True, corr_rx_warn=True, plt_warn=True, np_warn=True,
                              deprecated_warn=True, katcp_warn=True):
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
        # set the SPEAD2 logger to x only
        spead_logger = logging.getLogger('spead2')
        spead_logger.setLevel(logging.ERROR)
    if corr_rx_warn:
        # set the corr_rx logger to x only
        corr_rx_logger = logging.getLogger("corr2.corr_rx")
        corr_rx_logger.setLevel(logging.ERROR)
    if katcp_warn:
        # Set katcp.inspect_client logger to x messages
        import katcp
        import tornado
        katcp_logger = logging.getLogger('katcp.inspect_client')
        katcp_logger.setLevel(logging.FATAL)
        tornado_logger = logging.getLogger('tornado.application')
        tornado_logger.setLevel(logging.FATAL)
    if plt_warn:
        # This function disable matplotlibs deprecation warnings
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

    def Bold(self, msg=None):
        return (self.BOLD + msg + self.END)

    def Underline(self, msg=None):
        return (self.UNDERLINE + msg + self.END)


Style = Text_Style()


@contextlib.contextmanager
def ignored(*exceptions):
    """
    Context manager to ignore specifed exceptions
    :param: Exception
    :rtype: None
    """
    try:
        yield
    except exceptions:
        pass


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


def deprogram_hosts(self, timeout=60):
    """Function that deprograms F and X Engines
    :param: Object
    :rtype: None
    """
    try:
        hosts = self.correlator.xhosts + self.correlator.fhosts
        threaded_fpga_function(hosts, timeout, 'deprogram')
        LOGGER.info('F/X-engines deprogrammed successfully .')
        return True
    except Exception:
        LOGGER.error('Failed to deprogram all connected hosts.')
        return False

def human_readable_ip(hex_ip):
    hex_ip = hex_ip[2:]
    return '.'.join([str(int(x + y, 16)) for x, y in zip(hex_ip[::2], hex_ip[1::2])])


def confirm_out_dest_ip(self):
    """Confirm is correlators output destination ip is the same as the one in config file
    :param: Object
    :rtype: Boolean
    """
    try:
        xhost = self.correlator.xhosts[randrange(len(self.correlator.xhosts))]
        int_ip = xhost.registers.gbe_iptx.read()['data']['reg']
        xhost_ip = inet_ntoa(pack(">L", int_ip))
        dest_ip = self.correlator.configd['xengine']['output_destination_ip']
        assert dest_ip == xhost_ip
        return True
    except Exception:
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
def RunTestWithTimeout(test_timeout=1, errmsg='Test Timed-out'):
    """
    Context manager to execute tests with a timeout
    :param: test_timeout : int
    :param: errmsg : str
    :rtype: None
    """
    try:
        with TestTimeout(test_timeout):
            yield
    except Exception:
        Aqf.failed(errmsg)
        Aqf.end(traceback=True)

def get_local_src_names(self):
    """
    Calculate the number of inputs depending on correlators objects number of antennas
    :param: Object
    :rtype: List"""
    try:
        self.assertIsInstance(self.correlator, corr2.fxcorrelator.FxCorrelator)
    except AssertionError, e:
        errmsg = 'Failed to instantiate a correlator object: %s' % str(e)
        LOGGER.error(errmsg)
        Aqf.failed(errmsg)
        return
    else:
        return ['inp0{:02d}_{}'.format(x, i)
                for x in xrange(self.correlator.n_antennas)
                for i in 'xy']


def who_ran_test():
    """Get who ran the test."""
    try:
        Aqf.hop('Test ran by: {} on {} system on {}.\n'.format(getusername(),
                                                            os.uname()[1].upper(),
                                                            time.strftime("%Y-%m-%d %H:%M:%S")))
    except OSError:
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


class GracefullInterruptHandler(object):
    """
    Control-C keyboard interrupt handler.
    :param: Signal type
    :rtype: None
    """

    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False

        self.original_handler = signal.getsignal(self.sig)
        def handler(signum, frame):
            self.release()
            self.interrupted = True
        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, to):
        self.release()

    def release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
