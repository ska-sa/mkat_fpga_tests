from __future__ import division

import telnetlib,time
import os
import sys
import threading
import collections
import numpy as np
import pandas as pd
import csv
from corr2.utils import parse_ini_file
from nosekatreport import Aqf


test_conf = parse_ini_file('./config/test_conf.ini')
pdu_host_ips = test_conf['pdu_hosts']['pdu_ips'].split(',')
port = test_conf['pdu_hosts']['telnet_port']
user = test_conf['pdu_hosts']['username']
password = test_conf['pdu_hosts']['passwd']
pwr = 'phReading id:all power'
crnt = 'phReading id:all current'

def get_power_cons(host_IP, port=23, *args):
    def get_stdout(data, unit):
        return str([float(i.split()[1]) for i in data.splitlines()
                 if i.endswith(unit) if len(i.split()) == 3]).strip('[]')
    try:
        tn = telnetlib.Telnet(host_IP, port, timeout=5)
    except:
        return None
    else:
        try:
            _None = tn.read_until('User Name :', timeout=5)
            tn.write(user + '\r\n')
            _None = tn.read_until('Password  :', timeout=5)
            tn.write(password  + '\r\n')
            _None = tn.read_until('apc>', timeout=5)
            tn.write(pwr + '\r\n')
            stdout = tn.read_until('apc>', timeout=5)
            pwr_lst = get_stdout(stdout, 'kW')
            tn.write(crnt + '\r\n')
            stdout = tn.read_until('apc>', timeout=5)
            curr_lst = get_stdout(stdout, 'A')
            tn.close()
            smpl_time = str(int(time.time()))
            return [smpl_time, host_IP, curr_lst, pwr_lst]
        except:
            return None

def write_csv():
    with open('test.csv', 'wb') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        csv_writer.writerow(['Sample Time','PDU Host IP','Phase Current','Phase Power'])
        while True:
            for count, host in enumerate(pdu_host_ips, start=1):
                data = (get_power_cons(host, port, pwr, crnt))
                if data != None:
                    csv_writer.writerow(data)
                    csvfile.flush()
                print data

def get_ph_list(p_hosts, count=30):
    data_list = []
    while True:
        try:
            for pdu_host in pdu_hosts:
                data = (get_power_cons(pdu_host, port, pwr, crnt))
                data_list.append(data)
                print data
        except Exception:
            pass
        count -=1
        if count == 0: break
    return data_list
write_csv()
Ph_list = get_ph_list(pdu_host_ips,3)
import IPython;IPython.embed()
PhI_list = [map(float, PhI_val.values()[1].split(',')) for PhI_val in Ph_list]
Perc_Curr_drawn_Ph = [Ph1 / (Ph1 +  Ph2 + Ph3) for Ph1, Ph2, Ph3 in PhI_list]

max_min_curr_drawn = [np.max(PhI)/np.min(PhI) for PhI in PhI_list]
try:
    max_curr_drawn = np.max(max_min_curr_drawn)
except:
    raise Exception
max_rat = 1.33
Aqf.less(max_curr_drawn, max_rat,
    'The maximum load balance ratio {} should be < {}'.format(
    max_curr_drawn, max_rat))

max_volt = 220
tot_int_power = [max_volt * (Ph1 +  Ph2 + Ph3) for Ph1, Ph2, Ph3 in PhI_list]
average_power = np.mean(tot_int_power)

ave_power_per_rack = np.mean(np.sum([220*(Ph1 +  Ph2 + Ph3) for Ph1, Ph2, Ph3 in PhI_list]))
