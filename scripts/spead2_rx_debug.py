#!/usr/bin/env python
import atexit
import fabric.api as fab
import click
import logging
import coloredlogs
import sys

from corr2 import utils
from corr2.corr_rx import CorrRx
from corr2.dsimhost_fpga import FpgaDsimHost

corr_rx_logger = logging.getLogger("corr2.corr_rx")
spead2_logger = logging.getLogger("spead2")

@click.command()
@click.option('--config_file', default=None, help='Corr2 config file')
@click.option('--rx_port', default=8888, help='Which port the receiver will connect to!')
@click.option('--dsim_start', default=True, help='Enable and Start DEngine.')
@click.option('--capture_start', default=True, help='Start capture or not?')
@click.option('--debug', default=True, help='Ipython debug')
@click.option('--verbose', default=False, help='Debug verbosity')
def SpeadRx(config_file, rx_port, dsim_start, capture_start, debug, verbose):
    """
    Receive data from Correlator and play
    """
    if config_file is None:
        sys.exit("Usage: %s --help" % __file__)

    try:
        assert verbose
        _level = 'DEBUG'
        logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(pathname)s : '
                   '%(lineno)d - %(message)s')
        corr_rx_logger.setLevel(logging.DEBUG)
        spead2_logger.setLevel(logging.DEBUG)
        logging.debug('DEBUG MODE ENABLED')
    except:
        _level = 'INFO'
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(pathname)s : '
                   '%(lineno)d - %(message)s')
        corr_rx_logger.setLevel(logging.INFO)
        spead2_logger.setLevel(logging.INFO)
    finally:
        logger = logging.getLogger(__name__)
        coloredlogs.install(level=_level,logger=logger)


    if dsim_start and config_file:
        corr_conf = utils.parse_ini_file(config_file, ['dsimengine'])
        dsim_conf = corr_conf['dsimengine']
        dhost = FpgaDsimHost(dsim_conf['host'], config=dsim_conf)
        try:
            assert dhost.is_running()
            dhost.get_system_information()
        except:
            logger.error('DEngine Not Running!')

    def initialise_dsim(_dhost):
        logger.info('Reset digitiser simulator to all Zeros')
        try:
            _dhost.registers.flag_setup.write(adc_flag=0, ndiode_flag=0, load_flags='pulse')
        except Exception:
            logger.error('Failed to set _dhost flag registers.')
            pass
        try:
            for sin_source in _dhost.sine_sources:
                sin_source.set(frequency=0, scale=0)
                assert sin_source.frequency == sin_source.scale == 0
                try:
                    if sin_source.name != 'corr':
                        sin_source.set(repeat_n=0)
                except NotImplementedError:
                        logger.exception('Failed to reset repeat on sin_%s' %sin_source.name)
                logger.info('Digitiser simulator cw source %s reset to Zeros' %sin_source.name)
        except Exception:
            logger.error('Failed to reset sine sources on _dhost.')
            pass

        try:
            for noise_source in _dhost.noise_sources:
                noise_source.set(scale=0)
                assert noise_source.scale == 0
                logger.info('Digitiser simulator awg sources %s reset to Zeros' %noise_source.name)
        except Exception:
            logger.error('Failed to reset noise sources on _dhost.')
            pass

        try:
            for output in _dhost.outputs:
                output.select_output('signal')
                output.scale_output(1)
                logger.info('Digitiser simulator signal output %s selected.' %output.name)
        except Exception:
            logger.error('Failed to select output _dhost.')
            pass


    try:
        assert capture_start
    except AssertionError:
        @atexit.register
        def Cleanup():
            logger.info('baseline-correlation-products capture stopped!')
            receiver.stop()
            receiver.join()
            logger.info('Receiver stopped.')
            fab.local("kcpcmd -t 60 -s localhost:$(kcpcmd array-list | grep -a array-list | cut -f3 -d ' ' ) capture-stop baseline-correlation-products")
    else:
        logger.info('baseline-correlation-products capture started!')
        fab.local("kcpcmd -t 60 -s localhost:$(kcpcmd array-list | grep -a array-list | cut -f3 -d ' ' ) capture-start baseline-correlation-products")

    try:
        initialise_dsim(dhost)
        logger.info('Setting correlater noise by default')
        dhost.noise_sources['noise_corr'].set(0.0645)

        receiver = CorrRx(port=rx_port, queue_size=5)
        _multicast_ips =  corr_conf['xengine'].get('multicast_interface_address','239.100.0.1')
        # import IPython; IPython.embed(header='Python Debugger')
    except Exception as ex:
        template = "An exception of type {0} occured while trying to instantiate receiver. Arguments:\n{1!r}"
        message = template.format(type(ex), ex.args)
        logger.info(message)
    else:
        logger.info('Waiting for receiver to report running')
        receiver.daemon = True
        receiver.start(timeout=10)
        if receiver.running_event.wait(timeout=10):
            logger.info('Receiver ready')
        else:
            msg = 'Receiver not ready'
            logger.info(msg)
            raise RuntimeError(msg)
        try:
            raw_input('Press Enter get clean dump')
            dump = receiver.get_clean_dump()
        except KeyboardInterrupt:
            logger.info('Keyboard interrupt')
        except Exception:
            raise
        else:
            logger.info('Dump received')
        if debug:
            corr_rx_logger.setLevel(logging.FATAL)
            spead2_logger.setLevel(logging.FATAL)
            import IPython; IPython.embed(header='Python Debugger')


if __name__ == '__main__':
    SpeadRx()