#!/usr/bin/python
import os
import corr2
import time

config = os.environ['CORR2INI']
corr_fix = corr2.fxcorrelator.FxCorrelator('fxcorr', config_source=config)
corr_fix.initialise(program=False)
corr_fix.est_synch_epoch()
curr_epoch = corr_fix._synchronisation_epoch
print 'Reference synchronisation epoch: {}'.format(curr_epoch)
while (1):
    corr_fix.est_synch_epoch()
    epoch = corr_fix._synchronisation_epoch
    print 'Synch epoch: {}'.format(epoch)
    print 'Drift = {}ms'.format(1000*(curr_epoch-epoch))
    time.sleep(60)


