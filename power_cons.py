from __future__ import division

import telnetlib
import os
import sys
import threading
import numpy as np
import pandas as pd
from corr2.utils import parse_ini_file
from nosekatreport import Aqf

test_conf = parse_ini_file('./config/test_conf.ini')

test_config = test_conf['pdu_hosts']
pdu_host_ips = test_config['pdu_ips'].split(',')
port = test_config['port']
user = test_config['username']
password = test_config['passwd']
pwr = 'phReading all power'
crnt = 'phReading all current'

def get_power_cons(host_IP, port, *args):
    def get_stdout(data, unit):
        return str([float(i.split()[1]) for i in data.splitlines()
                 if i.endswith(unit) if len(i.split()) == 3]).strip('[]')
    tn = telnetlib.Telnet(host_IP, port, timeout=50)
    _None = tn.read_until('User Name :', timeout=50)
    tn.write(user + '\r\n')
    try:
        _None = tn.read_until('kat\r\nPassword  :', timeout=50)
    except EOFError:
        return False
    tn.write(password  + '\r\n')
    _None = tn.read_until('apc>', timeout=50)
    tn.write(pwr + '\r\n')
    stdout = tn.read_until('apc>', timeout=50)
    pwr_lst = get_stdout(stdout, 'kW')
    tn.write(crnt + '\r\n')
    stdout = tn.read_until('apc>', timeout=50)
    curr_lst = get_stdout(stdout, 'A')
    tn.close()
    return {'1. PDU_host': host_IP,
            '2. Power (kW)': pwr_lst,
            '3. Current (A)': curr_lst
            }
def dataframe_csv():
    test = []
    for count, host in enumerate(pdu_host_ips, start=1):
        test.append(get_power_cons(host, port, pwr, crnt))
    dataframe = pandas.DataFrame(data=test)
    dataframe.to_csv('test.csv', sep='\t', encoding='utf-8')

def get_ph_list(pdu_host):
    data_list = []
    count = 30
    while True:
        try:
            data_list.append(get_power_cons(pdu_host, port, pwr, crnt))
        except Exception:
            pass
        count -=1
        if count == 0: break
    return data_list

Ph_list = get_ph_list(pdu_host_ips[0])
PhI_list = [map(float, PhI_val.values()[1].split(',')) for PhI_val in Ph_list]
Perc_Curr_drawn_Ph = [Ph1 / (Ph1 +  Ph2 + Ph3) for Ph1, Ph2, Ph3 in PhI_list]

max_min_curr_drawn = [np.max(PhI)/np.min(PhI) for PhI in PhI_list]
max_curr_drawn = np.max(max_min_curr_drawn)
max_rat = 1.33
Aqf.less(max_curr_drawn, max_rat,
    'The maximum load balance ratio {} should be < {}'.format(
    max_curr_drawn, max_rat))

max_volt = 220
tot_int_power = [max_volt * (Ph1 +  Ph2 + Ph3) for Ph1, Ph2, Ph3 in PhI_list]
average_power = np.mean(tot_int_power)

ave_power_per_rack = np.mean(np.sum([220*(Ph1 +  Ph2 + Ph3) for Ph1, Ph2, Ph3 in PhI_list]))
