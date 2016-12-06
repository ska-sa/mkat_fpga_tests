#!/usr/bin/python
import os
import logging
import time
import casperfpga
import corr2

from casperfpga import katcp_fpga
from corr2 import fxcorrelator
from corr2 import utils
from corr2.dsimhost_fpga import FpgaDsimHost
from katcp.testutils import start_thread_with_cleanup
from corr2.corr_rx import CorrRx

dump_timeout = 10
logging.basicConfig(filename='debug_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)-7s - %(module)-8s - %(message)s')
logger = logging.getLogger(__name__)

#config = os.environ['CORR2INI']
#print 'Config file used = {}'.format(config)

#corr_conf = utils.parse_ini_file(config, ['dsimengine'])
#dsim_conf = corr_conf['dsimengine']
#dig_host = dsim_conf['host']

#dhost = FpgaDsimHost(dig_host, config=dsim_conf)
#if dhost.is_running():
#    dhost.get_system_information()
#    print 'Dsim is running'

#config_link = '/etc/corr/array0-bc8n856M4k'
#config_link2 = '/etc/corr/array0-bc8n856M32k'
#config_link3 = '/etc/corr/templates/bc8n856M4k'
#config_link4 = '/etc/corr/templates/bc8n856M32k'
#if os.path.exists(config_link):
#    correlator = fxcorrelator.FxCorrelator('rts correlator', config_source=config_link)
#elif os.path.exists(config_link2):
#    correlator = fxcorrelator.FxCorrelator('rts correlator', config_source=config_link2)
#elif os.path.exists(config_link3):
#    correlator = fxcorrelator.FxCorrelator('rts correlator', config_source=config_link3)
#else:
#    correlator = fxcorrelator.FxCorrelator('rts correlator', config_source=config_link4)
#correlator = fxcorrelator.FxCorrelator('rts correlator', config_source=config)

#correlator.initialise(program=False)
#f_engines = {f.host: f for f in correlator.fhosts}
#x_engines = {x.host: x for x in correlator.xhosts}
#for fpga in correlator.fhosts + correlator.xhosts:
#    if fpga.is_running():
#        fpga.get_system_information()

#print 'correlator is running'
#f = correlator.fhosts[0]
#fhost = correlator.fhosts[0]

#xhost = correlator.xhosts[0]
#x = correlator.xhosts[0]
try:
    receiver = CorrRx(port=8888, queue_size=5)
    corr_rx_logger = logging.getLogger("corr2.corr_rx")
    corr_rx_logger.setLevel(logging.DEBUG)
    spead2_logger = logging.getLogger("spead2")
    spead2_logger.setLevel(logging.DEBUG)
except Exception as ex:
    template = "An exception of type {0} occured while trying to instantiate receiver. Arguments:\n{1!r}"
    message = template.format(type(ex), ex.args)
    print message
else:
    try:
        #start_thread_with_cleanup(receiver, start_timeout=1)
        print('Waiting for receiver to report running')
        boop = receiver.start(timeout=10)
        print boop
        if receiver.running_event.wait(timeout=10):
            print('Receiver ready')
        else:
            print('Receiver not ready')
            raise
        raw_input('press to get clean dump')
        try:
            dump = receiver.get_clean_dump(dump_timeout=dump_timeout, discard=0)
        except KeyboardInterrupt:
            raise
        except:
            raise
        else:
            prev_ts = dump['timestamp'].value
            while True:
                try:
                    dump = receiver.get_clean_dump(dump_timeout=dump_timeout, discard=0)
                except KeyboardInterrupt:
                    raise
                except:
                    raise
                else:
                    ts = dump['timestamp'].value
                    sf = dump['scale_factor_timestamp'].value
                    ts_delta = ts-prev_ts
                    prev_ts = ts
                    print('Time delta between received dumps: {}'.format(ts_delta/sf))
    except KeyboardInterrupt:
        print '\nKeyboard interrupt detected... closing receiver.'
        pass
    except Exception as ex:
        template = "An exception of type {0} occured. Arguments:\n{1!r}"
        message = template.format(type(ex), ex.args)
        print message
        logger.error(message)
    #import IPython;IPython.embed()
    receiver.stop()
    receiver.join()
    print 'Receiver stopped.'
