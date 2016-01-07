import collections
import h5py
import numpy as np
import time

from nosekatreport import Aqf, aqf_vr
from casperfpga.utils import threaded_fpga_operation
from mkat_fpga_tests import correlator_fixture


VACC_FULL_RANGE = float(2**31)      # Max range of the integers coming out of VACC

def complexise(input_data):
    """Convert input data shape (X,2) to complex shape (X)"""
    return input_data[:,0] + input_data[:,1]*1j

def magnetise(input_data):
    id_c = complexise(input_data)
    id_m = np.abs(id_c)
    return id_m

def normalise(input_data):
    return input_data / VACC_FULL_RANGE

def normalised_magnitude(input_data):
    return normalise(magnetise(input_data))

def loggerise(data, dynamic_range=70, normalise_to=None):
    log_data = 10*np.log10(data)
    if normalise_to:
        max_log = normalise_to
    else:
        max_log = np.max(log_data)
    min_log_clip = max_log - dynamic_range
    log_data[log_data < min_log_clip] = min_log_clip
    return log_data

def baseline_checker(xeng_raw, check_fn):
    """Apply a test function to correlator data one baseline at a time

    Returns a set of all the baseline indices for which the test matches
    """
    baselines = set()
    for bl in range(xeng_raw.shape[1]):
        if check_fn(xeng_raw[:, bl, :]):
            baselines.add(bl)
    return baselines

def zero_baselines(xeng_raw):
    """Return baseline indices that have all-zero data"""
    return baseline_checker(xeng_raw, lambda bldata: np.all(bldata == 0))

def nonzero_baselines(xeng_raw):
    """Return baseline indices that have some non-zero data"""
    return baseline_checker(xeng_raw, lambda bldata: np.any(bldata != 0))

def all_nonzero_baselines(xeng_raw):
    """Return baseline indices that have all non-zero data"""
    return baseline_checker(xeng_raw, lambda bldata: np.all(np.linalg.norm(
                    bldata.astype(np.float64), axis=1) != 0))

def init_dsim_sources(dhost):
    """Select dsim signal output, zero all sources, output scalings to 0.5

    Also clear noise diode and adc overrange flags
    """
    # Reset flags
    dhost.registers.flag_setup.write(adc_flag=0, ndiode_flag=0,
                                     load_flags='pulse')
    for sin_source in dhost.sine_sources:
        sin_source.set(frequency=0, scale=0)
        try:
            sin_source.set(repeatN=0)
        except NotImplementedError:
            pass
    for noise_source in dhost.noise_sources:
        noise_source.set(scale=0)
    for output in dhost.outputs:
        output.select_output('signal')
        output.scale_output(0.5)

class CorrelatorFrequencyInfo(object):
    """Derrive various bits of correlator frequency info using correlator config"""

    def __init__(self, corr_config):
        """Initialise the class

        Parameters
        ==========
        corr_config : dict
            Correlator config dict as in :attr:`corr2.fxcorrelator.FxCorrelator.configd`

        """
        self.corr_config = corr_config
        self.n_chans = int(corr_config['fengine']['n_chans'])
        "Number of frequency channels"
        self.bandwidth = float(corr_config['fengine']['bandwidth'])
        "Correlator bandwidth"
        self.delta_f = self.bandwidth / self.n_chans
        "Spacing between frequency channels"
        f_start = 0. # Center freq of the first bin
        self.chan_freqs = f_start + np.arange(self.n_chans)*self.delta_f
        "Channel centre frequencies"
        self.sample_freq = float(corr_config['FxCorrelator']['sample_rate_hz'])
        self.sample_period = 1 / self.sample_freq
        self.fft_period = self.sample_period*2*self.n_chans
        """Time length of a single FFT"""

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
        assert samples_per_chan > 0
        assert chans_around > 0
        assert 0 <= chan < self.n_chans
        assert 0 <= chan + chans_around < self.n_chans
        assert 0 <= chan - chans_around < self.n_chans

        fc = self.chan_freqs[chan]
        start_chan = chan - chans_around
        end_chan = chan + chans_around
        if samples_per_chan == 1:
            return self.chan_freqs[start_chan:end_chan+1]

        start_freq = self.chan_freqs[start_chan] - self.delta_f/2
        end_freq = self.chan_freqs[end_chan] + self.delta_f/2
        sample_spacing = self.delta_f / (samples_per_chan - 1)
        num_samples = int(np.round(
            (end_freq - start_freq) / sample_spacing)) + 1
        return np.linspace(start_freq, end_freq, num_samples)

def get_dsim_source_info(dsim):
    """Return a dict with all the current sine, noise and ouput settings of a dsim"""
    info = dict(sin_sources={}, noise_sources={}, outputs={})
    for sin_src in dsim.sine_sources:
        info['sin_sources'][sin_src.name] = dict(scale=sin_src.scale,
                                                 frequency=sin_src.frequency)
    for noise_src in dsim.noise_sources:
        info['noise_sources'][noise_src.name] = dict(scale=noise_src.scale)

    for output in dsim.outputs:
        info['outputs'][output.name] = dict(output_type=output.output_type,
                                            scale=output.scale)
    return info

def iterate_recursive_dict(d, keys=()):
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
    if isinstance(d, collections.Mapping):
        for k in d:
            for rv in iterate_recursive_dict(d[k], keys + (k, )):
                yield rv
    else:
        yield (keys, d)

vstr = h5py.special_dtype(vlen=bytes)

class TestDataH5(object):
    """Save correlator dumps, source info and freeform snapshot info to hdf5 file"""
    def __init__(self, filename):
        self.h5 = h5py.File(filename, 'w')
        self.results_idx = 0

    def close(self):
        self.h5.close()

    def create_dataset_from_spead_item(self, ig, item_name, h5_name):
        item = ig.get_item(item_name)
        shape = ig[item_name].shape if item.shape == -1 else item.shape
        h5_shape = [1] + ([] if list(shape) == [1] else list(shape))
        h5_maxshape = [None] + h5_shape[1:]
        dtype = np.dtype(type(ig[item_name])) if shape == [] else item.dtype
        if dtype is None:
            dtype = ig[item_name].dtype
        self.h5.create_dataset(h5_name, h5_shape, maxshape=h5_maxshape, dtype=dtype)

    def create_dataset_from_value(self, value, h5_name):
        if value is None:
            # Can't store None's
            return
        if isinstance(value, list):
            value = np.array(value)
        if hasattr(value, 'dtype'):
            dtype = value.dtype
        elif isinstance(value, basestring):
            dtype = vstr
        else:
            dtype = np.dtype(type(value))
        shape = list(getattr(value, 'shape', []))
        if not self.results_idx:
            self.h5.create_dataset(
                h5_name, shape=[1] + shape, maxshape=[None] + shape, dtype=dtype)

    def add_value_to_h5(self, value, h5_name):
        if value is None:
            # Can't store None's
            return
        self.h5[h5_name].resize(self.results_idx + 1, axis=0)
        self.h5[h5_name][self.results_idx] = value

    def add_result(self, ig, source_info, snapblocks):
        # Assume each dump has the same data keys as the first
        for item_name in ig.keys():
            data = ig[item_name]
            h5_name = 'dumps/' + item_name
            if not self.results_idx:
                self.create_dataset_from_spead_item(ig, item_name, h5_name)
            self.add_value_to_h5(data, h5_name)

        # Add snapblocks, assuming they always have the same data structure
        for compound_key, value in iterate_recursive_dict(snapblocks):
            h5_path = 'snapblocks/' + '/'.join(compound_key)
            if not self.results_idx:
                self.create_dataset_from_value(value, h5_path)
            self.add_value_to_h5(value, h5_path)
        # Also that the source_info data structure remains unchanged
        for compound_key, value in iterate_recursive_dict(source_info):
            h5_path = 'source_info/' + '/'.join(compound_key)
            if not self.results_idx:
                self.create_dataset_from_value(value, h5_path)
            self.add_value_to_h5(value, h5_path)

        self.results_idx += 1


def get_feng_snapshots(feng_fpga, timeout=5):
    snaps = {}
    for snap in feng_fpga.snapshots:
        snaps[snap.name] = snap.read(
            man_valid=False, man_trig=False, timeout=timeout)
    return snaps

def get_snapshots(instrument):
    f_snaps = threaded_fpga_operation(instrument.fhosts, 25, (get_feng_snapshots, ))
    return dict(feng=f_snaps)

def get_source_object_and_index(instrument, input_name):
    """Return the DataSource object and local roach source index for a given input"""
    # Todo MM 2015-10-22
    # Check and fix the hardcoded stuffs
    source = [s['source'].name for s in instrument.fengine_sources
              if s['source'].name == input_name][0]
    source_index = 0 #[i for i, s in enumerate(source.host.data_sources)
                    #if s.name == source.name][0]
    return source, source_index

def set_coarse_delay(instrument, input_name, value=0):
    """ Sets coarse delay(default = 1) for Correlator baseline input.

        Parameters
            =========
            instrument
                Correlator object.
            input_name
                Baseline (eg.'m000_x').
            value
                Number of samples to delay
    """
    source, source_index = get_source_object_and_index(instrument, input_name)
    if source_index == 0:
        source.host.registers.coarse_delay0.write(coarse_delay=value)
        source.host.registers.tl_cd0_control0.write(arm='pulse', load_immediate=1)
    else:
        source.host.registers.coarse_delay1.write(coarse_delay=value)
        source.host.registers.tl_cd1_control0.write(arm='pulse', load_immediate=1)

def rearrange_snapblock(snap_data, reverse=False):
    segs = []
    for segment in sorted(snap_data.keys(), reverse=reverse):
        segs.append(snap_data[segment])
    return np.column_stack(segs).flatten()

def get_quant_snapshot(instrument, input_name, timeout=5):
    """Get the quantiser snapshot of named input. Snapshot will be assembled"""
    #import IPython;IPython.embed()

    # TODO MM 2015-10-22
    # Hardcoded shit. fix it
    host = [i['host'] for i in instrument.fengine_sources][0]
    source, source_index = ('m000_x', 0)#get_source_object_and_index(instrument, input_name)
    snap_name = 'snap_quant{}_ss'.format(source_index)
    snap = host.snapshots[snap_name] # source.host.snapshots[snap_name]
    snap_data = snap.read(
        man_valid=False, man_trig=False, timeout=timeout)['data']

    def get_part(qd, part):
        return {k: v for k,v in qd.items() if k.startswith(part)}
    real = rearrange_snapblock(get_part(snap_data, 'real'))
    imag = rearrange_snapblock(get_part(snap_data, 'imag'))
    quantiser_spectrum = real + 1j*imag
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
    bls_ordering = spead['bls_ordering']
    baseline_lookup = {tuple(bl): ind for ind, bl in enumerate(bls_ordering)}
    return baseline_lookup

def clear_all_delays(instrument, receiver):
    """Clears all delays on all fhosts.
    Param: Correlator object
    Return: None
    """
    delay_coefficients = ['0,0:0,0'] * len(instrument.fengine_sources)
    dump = receiver.get_clean_dump(10, discard=0)
    future_time = 200e-3
    dump_timestamp = (dump['sync_time'] + dump['timestamp'] /
                      dump['scale_factor_timestamp'])
    t_apply = (dump_timestamp + dump['int_time'] + future_time)
    reply = correlator_fixture.katcp_rct.req.delays(t_apply, *delay_coefficients)
    Aqf.is_true(reply.reply.reply_ok(), reply.reply.arguments[1])

def get_fftoverflow_qdrstatus(correlator):
    """Get dict of all roaches present in the correlator
    Param: Correlator object
    Return: Dict:
        Roach, QDR status, PFB counts
    """
    fhosts = {}
    xhosts = {}
    dicts = {}
    dicts['fhosts'] = {}
    dicts['xhosts'] = {}
    fengs = correlator.fhosts
    xengs = correlator.xhosts
    for fhost in fengs:
        fhosts[fhost.host] = {}
        fhosts[fhost.host]['QDR_okay'] = fhost.qdr_okay()
        for pfb, value in fhost.registers.pfb_ctrs.read()['data'].iteritems():
            fhosts[fhost.host][pfb] = value
        for xhost in xengs:
            xhosts[xhost.host] = {}
            xhosts[xhost.host]['QDR_okay'] = xhost.qdr_okay()
    dicts['fhosts'] = fhosts
    dicts['xhosts'] = xhosts
    return dicts

def check_fftoverflow_qdrstatus(correlator, last_pfb_counts):
    """Checks if FFT overflows and QDR status on roaches
    Param: Correlator object, last known pfb counts
    Return: list:
        Roaches with QDR status errors
    """
    QDR_error_roaches = set()
    fftoverflow_qdrstatus = get_fftoverflow_qdrstatus(correlator)
    curr_pfb_counts = get_pfb_counts(
        fftoverflow_qdrstatus['fhosts'].items())
    # Test FFT Overflow status
    [Aqf.equals(curr_pfb_value, last_pfb_value,
        "Check if the is no PFB FFT overflow on {}".format(curr_pfb_host))
    for (curr_pfb_host, curr_pfb_value), (curr_pfb_host, last_pfb_value) in zip(
        last_pfb_counts.items(), curr_pfb_counts.items())
        if curr_pfb_host is curr_pfb_host]

    # Test QDR error flags
    for hosts_status in fftoverflow_qdrstatus.values():
        for host, hosts_status in hosts_status.items():
            if hosts_status['QDR_okay'] is False:
                Aqf.step('QDR status on {} not Okay.'.format(host))
                QDR_error_roaches.add(host)
    # Test QDR status
    Aqf.is_false(QDR_error_roaches,
                 'Check for QDR errors.')
    return QDR_error_roaches

def get_vacc_offset(xeng_raw):
    """Assuming a tone was only put into input 0, figure out if VACC is roated by 1"""
    b0 = np.abs(complexise(xeng_raw.value[:, 0]))
    b1 = np.abs(complexise(xeng_raw.value[:, 1]))
    if np.max(b0) > 0 and np.max(b1) == 0:
        # We expect autocorr in baseline 0 to be nonzero if the vacc is
        # properly aligned, hence no offset
        return 0
    elif np.max(b1) > 0 and np.max(b0) == 0:
        return 1
    else:
        raise ValueError('Could not determine VACC offset')

def get_and_restore_initial_eqs(test_instance, correlator):
    initial_equalisations = {input: eq_info['eq'] for input, eq_info
                             in correlator.fops.eq_get().items()}

    def restore_initial_equalisations():
        for input, eq in initial_equalisations.items():
            correlator.fops.eq_set(source_name=input, new_eq=eq)

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
        pfb_list[host] = (pfb_value['pfb_of0_cnt'],
                          pfb_value['pfb_of1_cnt'])
    return pfb_list
