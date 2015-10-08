import collections
import h5py
import numpy as np

from casperfpga.utils import threaded_fpga_operation

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

def loggerise(data, dynamic_range=70):
    log_data = 10*np.log10(data)
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
    source = [s for s in instrument.fengine_sources if s.name == input_name][0]
    source_index = [i for i, s in enumerate(source.host.data_sources)
                    if s.name == source.name][0]
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
    source, source_index = get_source_object_and_index(instrument, input_name)
    snap_name = 'snap_quant{}_ss'.format(source_index)
    snap = source.host.snapshots[snap_name]
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
        baseline lookup
    """
    bls_ordering = spead['bls_ordering']
    baseline_lookup = {tuple(bl): ind for ind, bl in enumerate(bls_ordering)}
    return baseline_lookup
