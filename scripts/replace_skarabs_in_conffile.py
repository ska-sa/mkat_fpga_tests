#!/usr/bin/env python

# Brutal way of replacing fhosts,xhosts and dhost in template(s)
import argparse
import sys
import fileinput

def main():
    parser = argparse.ArgumentParser(description='Simplified way of replacing fhosts, xhosts and dhost '
                                                 'from config templates')
    parser.add_argument('-l','--leaf', help='Specify which LEAF contains the skarabs needed',
                        required=True)
    parser.add_argument('-f','--config', help='Specify which config file, look in /etc/corr/templates',
                        required=True)
    args = vars(parser.parse_args())


    _which_leaf = args.get('leaf', False)
    config = args.get("config", False)

    def get_skarabs_from_leaf(_which_leaf):
        dnsmasq_leases = '/var/lib/misc/dnsmasq.leases'
        with open(dnsmasq_leases) as f:
            dnsmasq_leases_data = f.readlines()
        skarabs = [host for i in dnsmasq_leases_data
                   if ('100.%s' %_which_leaf) in i for host in i.split()
                                       if host.startswith('skarab')]
        return skarabs
    if _which_leaf or config:
        skarab_list = sorted(get_skarabs_from_leaf(_which_leaf))
        fhosts = 'hosts = %s\n' % ','.join(skarab_list[:4])
        xhosts = 'hosts = %s\n' % ','.join(skarab_list[4:8])
        dhost = 'host = %s\n' % skarab_list[-1]

        for line in fileinput.input([config], inplace=True):
            if line.strip().startswith('hosts = '):
                line = fhosts
            if line.strip().startswith('hosts = '):
                line = xhosts
            if line.strip().startswith('host = '):
                line = dhost
            sys.stdout.write(line)


if __name__ == "__main__":
    main()